import mariadb
import pandas as pd
import datetime

# 데이터베이스 연결 정보
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "7496",
    "database": "EZEN",
    "local_infile": True  # 반드시 추가해야 함
}

#database 안에 있는 하나의 테이블의 데이터를 조회할 수 있다.
def show_data(table):
    #고정된 테이블을 사용하므로 유념해야 함
    table = 'finance'
    conn = None
    try:
        conn = mariadb.connect(**DB_CONFIG)
        cur = conn.cursor()

        # 데이터 조회
        cur.execute(f"SELECT * FROM {table}")
        columns = [x[0] for x in cur.description]
        #print(columns)
        #print(len(columns))

        rows = cur.fetchall()
        #print(str(rows[0]))
        result = []

        datetime_indexes = [i for i, value in enumerate(rows[0]) if isinstance(value, (datetime.date, datetime.datetime))]
        if datetime_indexes:
            datetime_columns = [(columns[i], i) for i in datetime_indexes]

        if rows != None:
            for r in rows:
                new_row = []
                for j, v in enumerate(r):
                    if j == 9:
                        val = v.strftime("%Y:%m:%d")
                        #print(val)
                    else:
                        val = v
                
                    new_row.append(val)
                result.append(new_row)
        
        result = pd.DataFrame(result, columns=columns)
        #print(result)
        #result.to_csv('./sql_into_csv_.csv')
        return result

    except mariadb.Error as e:
        print(f"❌ 데이터 조회 실패: {e}")

    finally:
        if conn:
            cur.close()
            conn.close()

# if __name__ == '__main__':
#     r = show_data('finance')