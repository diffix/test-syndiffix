import psycopg2
import pandas as pd
import sqlalchemy as sq
import hashlib
import re
import os

class queryHandler:
    ''' This is for determining the correct combs table to query from an
        SQL query
    '''
    def __init__(self, pgHost=None, pgUser=None, pgPass=None, dbName='sdx_demo', port=5432, doPrint=False):
        self.p = doPrint
        self.pgHost = pgHost
        self.dbName = dbName
        self.pgUser = pgUser
        self.pgPass = pgPass
        self.port = port
        if not self.pgHost:
            self.pgHost = os.environ['GDA_POSTGRES_HOST']
        if not self.pgUser:
            self.pgUser = os.environ['GDA_POSTGRES_USER']
        if not self.pgPass:
            self.pgPass = os.environ['GDA_POSTGRES_PASS']
        self.sio = sqlIo(self.pgHost, self.dbName, self.pgUser, self.pgPass, self.port, connect=True)
        self.cmd = combsMetaData(self.sio, doPrint=self.p)
        self.ct = combsTables(doPrint=self.p)

    def query(self, sqlOrig):
        sqlSyn = self.getSynSql(sqlOrig)
        # sqlOrig is the original sql for the original table,
        # and sqlSyn is the sql modified to work on the appropriate synthetic
        # table
        dfOrig = self.sio.querySqlDf(sqlOrig)
        dfSyn = self.sio.querySqlDf(sqlSyn)
        return dfOrig, dfSyn

    def getSynSql(self, sqlOrig):
        sqlCopy = str(sqlOrig)
        tableOrig = self._findTable(sqlOrig)
        # tableOrig is the table term in the original sql
        tableBase = tableOrig.replace('_orig_','')
        metaData = self.cmd.getMetaData(tableBase)
        sortedMetaData = self.ct.sortColsByLen(metaData)
        # sortedMetaData contains all possible columns, sorted from longest
        # to shortest, and then alphabetically
        queryColumns = []
        for column in sortedMetaData:
            if column in sqlCopy:
                queryColumns.append(column)
                sqlCopy = sqlCopy.replace(column, '')
        # queryColumns contains the columns from original sql query
        tableSyn = self.ct.makeTableFromColumns(queryColumns, tableBase)
        # tableSyn is the synthetic table we should be using
        sqlSyn = sqlOrig.replace(tableOrig, tableSyn)
        return sqlSyn

    def _findTable(self, sql):
        # Assume that there is one table in the query, it contains no
        # spaces, it ends with '_orig_', and there is only one such term
        # in the query
        terms = sql.split()
        for term in terms:
            if '_orig_' in term:
                return term

class sqlIo:
    def __init__(self, pgHost, dbName, pgUser, pgPass, port=5432, connect=True, doPrint=False):
        self.p = doPrint
        self.pgHost = pgHost
        self.dbName = dbName
        self.pgUser = pgUser
        self.pgPass = pgPass
        self.port = port
        if connect:
            self.connect()
            self.createEngine()

    def connect(self):
        connStr = str(
            f"host={self.pgHost} port={self.port} dbname={self.dbName} user={self.pgUser} password={self.pgPass}")
        self.con = psycopg2.connect(connStr)
        self.cur = self.con.cursor()

    def close(self):
        self.cur.close()
        self.con.close()

    def createEngine(self):
        self.engine = sq.create_engine(f"postgresql://{self.pgUser}:{self.pgPass}@{self.pgHost}:{self.port}/{self.dbName}")
        #engine = sq.create_engine(f'postgresql+psycopg2://{pgUser}:{pgPass}@{pgHost}:5432/{databaseName}')

    def loadDataFrame(self, df, tableName):
        if not self.tableExists(tableName, df.shape[0]):
            print(f"Loading table {tableName}")
            df.to_sql(tableName, self.engine)
        else:
            print(f"Table {tableName} already loaded, skipping")

    def tableExists(self, tableName, numRows):
        sql = f"SELECT EXISTS ( SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename  = '{tableName}' ) "
        ans = self.querySql(sql)
        if ans[0][0] is False:
            return False
        # table is there, so let's find out if all the columns are there
        sql = f"SELECT count(*) FROM {tableName}"
        ans = self.querySql(sql)
        print(ans)
        if ans[0][0] == numRows:
            return True
        sql = f"DROP TABLE IF EXISTS {tableName}"
        self.modSql(sql)
        return False

    def modSql(self, sql):
        self.executeSql(sql)
        self.con.commit()

    def executeSql(self, sql):
        if self.p: print(sql)
        try:
            self.cur.execute(sql)
        except Exception as err:
            print ("cur.execute() error:", err)
            print ("Exception TYPE:", type(err))
        else:
            pass
            if self.p: print("SQL ok")

    def querySql(self, sql):
        self.executeSql(sql)
        return self.cur.fetchall()

    def querySqlDf(self, sql):
        self.executeSql(sql)
        df = pd.DataFrame(self.cur.fetchall())
        df.columns = [desc[0] for desc in self.cur.description]
        return df

class combsMetaData:
    def __init__(self, sio, doPrint=False):
        self.p = doPrint
        self.sio = sio
        pass

    def getMetaData(self, tableBase):
        metaTableName = tableBase + '_meta'
        sql = f'''SELECT columnName FROM {metaTableName}'''
        ans = self.sio.querySql(sql)
        return [x[0] for x in ans]

    def putMetaData(self, tableBase, columns):
        metaTableName = tableBase + '_meta'
        sql = f"DROP TABLE IF EXISTS {metaTableName}"
        self.sio.modSql(sql)
        sql = f"CREATE TABLE {metaTableName} (columnName TEXT)"
        self.sio.modSql(sql)
        for column in columns:
            sql = f'''INSERT INTO {metaTableName} (columnName) VALUES('{column}')'''
            self.sio.modSql(sql)

class combsTables:
    def __init__(self, doPrint=False):
        self.p = doPrint
        pass

    def makeTableFromColumns(self, columns, tableBase):
        # order columns longest to shortest (this to avoid cases where a
        # shorter column is a substring of a longer column)
        if len(columns) == 0:
            return tableBase + '_syn_'
        sortedCols = self.sortColsByLen(columns)
        table = tableBase
        for col in sortedCols:
            # postgres allowed characters
            col = "".join([c if c.isalnum() else "_" for c in col])
            table += f"_{col}"
        # This is to deal with table names over the postgres limit.
        if len(table) > 60:
            alphaHash = self._alphanumeric_hash(table)
            table = table[:50] + alphaHash[:10]
        return table

    def _alphanumeric_hash(self, input_string):
        hash_object = hashlib.sha256(input_string.encode())
        hex_hash = hash_object.hexdigest()
        alphanumeric_hash = re.sub(r'\W+', '', hex_hash)
        return alphanumeric_hash

    def sortColsByLen(self, columns):
        columns.sort()
        colLen = []
        for column in columns:
            colLen.append([column, len(column)])
        colLen.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in colLen]

if __name__ == '__main__':
    "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema' limit 20;"
    qh = queryHandler(doPrint=True)
    "census_num_persons_worked_for_employer_migration_c708d72893b"
    sql = '''
        SELECT education, count(*) as cnt
        FROM census_orig_
        GROUP BY 1
    '''
    dfOrig, dfSyn = qh.query(sql)
    print(dfOrig)
    print(dfSyn)
    quit()
    sql = f'''
        SELECT "marital stat", "migration code-change in msa", "num persons worked for employer"
        FROM census_orig_
        LIMIT 5
    '''
    dfOrig, dfSyn = qh.query(sql)
    print(dfOrig)
    print(dfSyn)
    sql = f'''
        SELECT age, "migration code-move within reg", "capital gains"
        FROM census_orig_
        LIMIT 5
    '''
    dfOrig, dfSyn = qh.query(sql)
    print(dfOrig)
    print(dfSyn)

    sql = f'''
        SELECT *
        FROM census_orig_
        LIMIT 5
    '''
    dfOrig, dfSyn = qh.query(sql)
    print(dfOrig)
    print(dfSyn)
