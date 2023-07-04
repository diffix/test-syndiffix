import sys
import os
import fire
import pandas as pd
import numpy as np

from gretel_client import configure_session
from gretel_client.projects import create_or_get_unique_project
from gretel_client.projects.models import read_model_config
from gretel_client.helpers import poll

from ydata.sdk.synthesizers import RegularSynthesizer
from ydata_synthetic.synthesizers.regular import RegularSynthesizer as RegularSynthesizerLocal
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.csvUtils import readCsv


def resultCsvPath(csvPath, method):
    path = os.path.join(os.path.dirname(csvPath), f'../../results{method}', os.path.basename(csvPath) + 'result.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

class External(object):

    def ydata(self, csvPath, local=True):
        pd.set_option("max_colwidth", None)

        df = readCsv(csvPath)
        nRows = df.shape[0]

        try:
            if local:
                synth = RegularSynthesizerLocal(modelname='dragan', model_parameters=ModelParameters(), n_discriminator=3)
                train_args = TrainParameters(epochs=5, sample_interval=100)
                num_cols = list(df.select_dtypes([np.number]).columns)
                cat_cols = list(df.select_dtypes(include='object').columns)
                synth.fit(data=df, train_arguments = train_args, num_cols = num_cols, cat_cols = cat_cols)
            else:
                synth = RegularSynthesizer()
                synth.fit(df)

            syntheticDf = synth.sample(n_samples=nRows)
        except Exception as e:
            print(e, 'encountered when processing', csvPath)
            return None

        syntheticDf.to_csv(resultCsvPath(csvPath, 'ydata'))
        print(syntheticDf.round(3))

    def gretel(self, csvPath):
        pd.set_option("max_colwidth", None)

        configure_session(api_key=os.environ['GRETEL_API_KEY'], cache="yes", validate=True)

        project = create_or_get_unique_project(name="synthetic-data")
        config = read_model_config("synthetics/tabular-lstm")
        config["models"][0]["synthetics"]["params"]["epochs"] = 10
        config["models"][0]["synthetics"]["generate"]["num_records"] = 10
        model = project.create_model_obj(model_config=config, data_source=csvPath)

        try:
            model.submit_cloud()
            poll(model)
            previewDf = readCsv(model.get_artifact_link("data_preview"), compression="gzip")
            print(previewDf)

            nRows = readCsv(csvPath).shape[0]
            recordHandler = model.create_record_handler_obj(
                params={"num_records": nRows, "max_invalid": None}
            )
            recordHandler.submit_cloud()
            poll(recordHandler)

            syntheticDf = readCsv(recordHandler.get_artifact_link("data"), compression="gzip")
        except Exception as e:
            print(e, 'encountered when processing', csvPath)
            return None

        syntheticDf.to_csv(resultCsvPath(csvPath, 'gretel'))

        print(syntheticDf.round(3))

    def many(self, *csvPaths, onlyMissing=False, method=None):
        if not method:
            "Method argument required"

        for i, csvPath in enumerate(csvPaths):
            print(i + 1, ' / ', len(csvPaths), csvPath)
            if onlyMissing and os.path.isfile(resultCsvPath(csvPath, method)):
                print('Result file exists, skipping')
                continue
            if method == 'ydata':
                self.ydata(csvPath)
            if method == 'gretel':
                self.gretel(csvPath)


if __name__ == "__main__":
    fire.Fire(External)
