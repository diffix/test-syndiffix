import psycopg2

class sqlIo:
    def __init__(self, pgHost, dbName, pgUser, pgPass, port=5432):
        connStr = str(
            f"host={pgHost} port={5432} dbname={dbName} user={pgUser} password={pgPass}")
        self.con = psycopg2.connect(connStr)
        self.cur = self.con.cursor()

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
        print(ans)
        quit()
        pass

    def putMetaData(self, tableBase, columns):
        metaTableName = tableBase + '_meta'
        sql = f"DROP TABLE IF EXISTS {metaTableName}"
        self.sio.modSql(sql)
        sql = f"CREATE TABLE {metaTableName} (columnName TEXT)"
        self.sio.modSql(sql)
        for column in columns:
            sql = f'''INSERT INTO {metaTableName} VALUES("{column}")'''
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
            # This generates a (most likely) unique string with only
            # postgres allowed characters
            col = "".join([c if c.isalnum() else "_" for c in col])
            table += f"_{col}"
        return table

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