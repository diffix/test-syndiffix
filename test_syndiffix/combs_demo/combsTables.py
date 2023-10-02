import psycopg2
import sqlalchemy as sq
import hashlib
import re

class QueryHandler:
    def __init__(self, pgHost=None, pgUser=None, pgPass=None, dbName='sdx_demo', port=5432):
        pass

class sqlIo:
    def __init__(self, pgHost, dbName, pgUser, pgPass, port=5432, connect=True):
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
        print(sql)
        try:
            self.cur.execute(sql)
        except Exception as err:
            print ("cur.execute() error:", err)
            print ("Exception TYPE:", type(err))
        else:
            print("SQL ok")

    def querySql(self, sql):
        self.executeSql(sql)
        return self.cur.fetchall()

class combsMetaData:
    def __init__(self, sio):
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
    def __init__(self):
        pass

    def makeTableFromColumns(self, columns, tableBase):
        # order columns longest to shortest (this to avoid cases where a
        # shorter column is a substring of a longer column)
        sortedCols = self._sortColsByLen(columns)
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

    def _sortColsByLen(self, columns):
        columns.sort()
        colLen = []
        for column in columns:
            colLen.append([column, len(column)])
        colLen.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in colLen]

if __name__ == '__main__':
    ct = combsTables()
    table = ct.makeTableFromColumns(['a', 'b', 'c ba', 'ba c'], 'taxi')
    print(table)
    pass