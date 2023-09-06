import json
import sys
import os
import fire
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.csvUtils import readCsv


def resultCsvPath(csvPath, method):
    path = os.path.join(os.path.dirname(csvPath), f'../../results{method}', os.path.basename(csvPath))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


class External(object):
    """
    Tool to generate synthetic data using external APIs for comparison.

    NOTE: Not all of these work as expected. `local=True` seem to require GPU to be available to make sense.
    """

    def __init__(self) -> None:
        self._gretelModelList = None

    def tonic(self, csvPath):
        from tonic_api.api import TonicApi

        apiToken = os.environ['TONIC_API_TOKEN']
        workspaceId = os.environ['TONIC_WORKSPACE']

        tonic = TonicApi("https://app.tonic.ai", apiToken)

        # test_data_science
        workspace = tonic.get_workspace(workspaceId)
        models = list(workspace.models)

        modelName = os.path.splitext(os.path.basename(csvPath))[0].replace('-', '_').lower()
        df = readCsv(csvPath)
        nRows = df.shape[0]
        print("available models:", [m.name for m in models], "requested model name:", modelName)

        model = next(m for m in models if m.name == modelName)
        trainedModel = workspace.get_most_recent_trained_model_by_model_id(model.id)
        print(trainedModel.describe())
        try:
            syntheticDf = trainedModel.sample(nRows)
        except Exception as e:
            print(e, 'encountered when processing', csvPath)
            return None

        syntheticDf.to_csv(resultCsvPath(csvPath, 'tonic'),
                           index=False, sep=',', header=True)
        print(syntheticDf.round(3))

    def ydata(self, csvPath, local=True):
        pd.set_option("max_colwidth", None)

        df = readCsv(csvPath)
        nRows = df.shape[0]

        try:
            if local:
                from ydata_synthetic.synthesizers.regular import RegularSynthesizer as RegularSynthesizerLocal
                from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

                synth = RegularSynthesizerLocal(
                    modelname='dragan', model_parameters=ModelParameters(), n_discriminator=3)
                train_args = TrainParameters(epochs=5, sample_interval=100)
                num_cols = list(df.select_dtypes([np.number]).columns)
                cat_cols = list(df.select_dtypes(include='object').columns)
                synth.fit(data=df, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)
            else:
                from ydata.sdk.synthesizers import RegularSynthesizer
                synth = RegularSynthesizer()
                synth.fit(df)

            syntheticDf = synth.sample(n_samples=nRows)
        except Exception as e:
            print(e, 'encountered when processing', csvPath)
            return None

        syntheticDf.to_csv(resultCsvPath(csvPath, 'ydata' + ('_local' if local else '')),
                           index=False, sep=',', header=True)
        print(syntheticDf.round(3))

    def gretel(self, csvPath, local=False):
        pd.set_option("max_colwidth", None)

        if local:
            from gretel_synthetics.actgan import ACTGAN

            df = readCsv(csvPath)
            nRows = df.shape[0]
            model = ACTGAN(
                verbose=True,
                auto_transform_datetimes=True,
            )

            model.fit(df)
            syntheticDf = model.sample(nRows)
        else:
            from gretel_client import configure_session
            from gretel_client.projects import create_or_get_unique_project
            from gretel_client.projects.models import read_model_config
            from gretel_client.helpers import poll

            configure_session(api_key=os.environ['GRETEL_API_KEY'], cache="yes", validate=True)

            project = create_or_get_unique_project(name="synthetic-data")
            config = read_model_config("synthetics/tabular-lstm")
            config["models"][0]["synthetics"]["generate"]["num_records"] = 10
            model = project.create_model_obj(model_config=config, data_source=csvPath)

            try:
                model.submit_cloud()
                poll(model)
                previewDf = readCsv(model.get_artifact_link("data_preview"), compression="gzip")
                print(previewDf)

                nRows = readCsv(csvPath).shape[0]
                recordHandler = model.create_record_handler_obj(
                    params={"num_records": nRows, "max_invalid": nRows * 3}
                )
                recordHandler.submit_cloud()
                poll(recordHandler)

                syntheticDf = readCsv(recordHandler.get_artifact_link("data"), compression="gzip")
            except Exception as e:
                print(e, 'encountered when processing', csvPath)
                return None
            finally:
                with open(resultCsvPath(csvPath, 'gretel') + '.log.json', 'w') as f:
                    json.dump(model.logs, f)
        syntheticDf.to_csv(resultCsvPath(csvPath, 'gretel' + ('_local' if local else '')),
                           index=False, sep=',', header=True)

        print(syntheticDf.round(3))

    def gretelDlLogs(self, csvPath):
        """
        Download the logs for completed `gretel` synthetization runs.

        Assumes the project is called `synthetic-data` as obtained from the `gretel` command.
        """
        from gretel_client import configure_session
        from gretel_client.projects.jobs import Status
        from gretel_client.projects import create_or_get_unique_project

        if self._gretelModelList is None:
            # This API call takes a while so catching for the sake of `many` command.
            configure_session(api_key=os.environ['GRETEL_API_KEY'], cache="yes", validate=True)
            project = create_or_get_unique_project(name="synthetic-data")
            self._gretelModelList = list(project.search_models(limit=12341234))

        csvFile = os.path.basename(csvPath)
        matching = [m for m in self._gretelModelList if csvFile in m.data_source and m.status == Status.COMPLETED]
        if matching:
            mostRecent = matching[-1]
            with open(resultCsvPath(csvPath, 'gretel') + '.log.json', 'w') as f:
                json.dump(mostRecent.logs, f)
        else:
            print("model for", csvPath, "not found in Gretel API")

    def many(self, *csvPaths, onlyMissing=False, method=None, pick=None, logs=False):
        """
        Process many CSV files using a chosen `method`.

        - onlyMissing - if True will only process input files for which the output files are missing
        - pick - process only the nth position of the input `csvPaths` list
        - logs - if True will download logs instead of processing (currently applies to gretel only)
        """
        if method is None:
            raise ValueError("Method argument required")

        for i, csvPath in enumerate(csvPaths):
            if pick is not None and pick > len(csvPaths):
                raise ValueError("--pick out of range")
            if pick is not None and pick != i:
                continue
            print(i + 1, ' / ', len(csvPaths), csvPath)
            if onlyMissing:
                if logs:
                    if os.path.isfile(resultCsvPath(csvPath, method) + '.log.json'):
                        print('Result log file exists, skipping')
                        continue
                else:
                    if os.path.isfile(resultCsvPath(csvPath, method)):
                        print('Result file exists, skipping')
                        continue
            if method == 'ydata':
                self.ydata(csvPath)
            if method == 'gretel':
                if logs:
                    self.gretelDlLogs(csvPath)
                else:
                    self.gretel(csvPath)


if __name__ == "__main__":
    fire.Fire(External)
