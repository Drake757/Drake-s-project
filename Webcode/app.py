from flask import Flask, render_template, request,session,url_for,redirect
import pymysql
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

app = Flask(__name__)
app.secret_key = "123456"


@app.route('/', methods=['GET', 'POST'])
def mainpage():
    return render_template('mainpage.html',x=0)

@app.route('/returnback', methods=['GET', 'POST'])
def returnfuc():
    return render_template('mainpage.html')

@app.route('/signin',methods=['GET', 'POST'])
def signin():
    return render_template('signin.html')

@app.route('/login',methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/registuser', methods=['GET', 'POST'])
def registuser():
    username = request.form.get('username')
    password = request.form.get('password2')
    version1 = request.form.get('version1')
    version2 = request.form.get('version2')
    print(version1)
    conn = pymysql.connect(host='127.0.0.1', user='root', password='hushanxin.', db='RUNOOB', port=3306, charset='utf8')
    cur = conn.cursor()
    if version1=='1':
        sql = "INSERT INTO students_tbl (username,password,version)VALUES (%s,%s,%s)"
        try:
            cur.execute(sql, (username, password,version1))
            conn.commit()
            cur.close()
            return render_template('login.html')
        except:
            conn.rollback()
    if version2=='2':
        sql = "INSERT INTO students_tbl (username,password,version)VALUES (%s,%s,%s)"
        try:
            cur.execute(sql, (username, password,version2))
            conn.commit()
            cur.close()
            return render_template('login.html')
        except:
            conn.rollback()
    conn.close()

@app.route('/LOGIN', methods=['GET', 'POST'])
def check():
        username = request.form.get('username')
        password = request.form.get('password')
        session['session_username'] =username
        conn = pymysql.connect(host='127.0.0.1', user='root', password='hushanxin.', db='RUNOOB', port=3306,
                               charset='utf8')
        cur = conn.cursor()
        sql2 = "select * from students_tbl where username = %s and password = %s"
        try:
            cur.execute(sql2, (username, password))
            results = cur.fetchall()
            session['version'] = results[0][3]
            if len(results) >= 1 and session['version']==2:
                return render_template('mainpage.html',x=session['version'],name=results[0][1])
            if len(results) >= 1 and session['version'] == 1:
                sql3 = "select * from project"
                try:
                    cur.execute(sql3)
                    table = cur.fetchall()
                    df = pd.DataFrame(table, dtype=float)
                    print(df[0][0])
                    student_id = []
                    sex=[]
                    Medu=[]
                    Fedu=[]
                    failures=[]
                    study_time=[]
                    paid=[]
                    higher=[]
                    freetime=[]
                    goout=[]
                    absences=[]
                    G1=[]
                    G2=[]
                    G3=[]
                    for i in range(0,394):
                        student_id.append(i)
                        sex.append(df[0][i])
                        Medu.append(df[1][i])
                        Fedu.append(df[2][i])
                        failures.append(df[3][i])
                        study_time.append(df[4][i])
                        paid.append(df[5][i])
                        higher.append(df[6][i])
                        freetime.append(df[7][i])
                        goout.append(df[8][i])
                        absences.append(df[9][i])
                        G1.append(df[10][i])
                        G2.append(df[11][i])
                        G3.append(df[12][i])
                    print(student_id)
                    return render_template('mainpage.html', x=session['version'])
                except:
                    conn.rollback()
                    return render_template('login.html')
            else:
                return 'The user name or password is incorrect'
            conn.commit()

        except:
            conn.rollback()
            return render_template('login.html')

        conn.close()

@app.route('/display', methods=['GET', 'POST'])
def display():
    conn = pymysql.connect(host='127.0.0.1', user='root', password='hushanxin.', db='RUNOOB', port=3306,
                           charset='utf8')
    cur = conn.cursor()
    sql3 = "select * from project where student_id = %s "
    sql4 = "select * from user_table where user_name = %s "
    try:
        cur.execute(sql4, (session['session_username']))
        name = cur.fetchall()
        print(name[0][0])
        cur.execute(sql3, (name[0][0]))
        id = cur.fetchall()
        if id[0][0]== name[0][0]:
            return render_template('display.html',x=id[0][11],y=id[0][12],z=id[0][13])
        else:
            return render_template('mainpage.html')
        conn.commit()

    except:
        conn.rollback()
        return render_template('mainpage.html')

    conn.close()


@app.route('/comparison', methods=['GET', 'POST'])
def comparison():
    conn = pymysql.connect(host='127.0.0.1', user='root', password='hushanxin.', db='RUNOOB', port=3306,
                           charset='utf8')
    cur = conn.cursor()
    sql5 = "select G1 from project "
    sql6 = "select G2 from project "
    sql7 = "select G3 from project "
    sql8 = "select * from user_table where user_name = %s "
    sql9 = "select * from project where student_id = %s "
    try:
        cur.execute(sql5)
        G1 = cur.fetchall()
        G1total=0
        G2total=0
        G3total=0
        # G1LIST =[]
        # G2LIST = []
        # G3LIST = []
        for i in range(0, 395):
            # G1LIST.append(G1[i][0])
            G1total= G1total+G1[i][0]
        G1average=G1total/395
        cur.execute(sql6)
        G2 = cur.fetchall()
        for i in range(0, 395):
            # G2LIST.append(G2[i][0])
            G2total = G2total + G2[i][0]
        G2average=G2total/395
        cur.execute(sql7)
        G3 = cur.fetchall()
        for i in range(0, 395):
            # G3LIST.append(G3[i][0])
            G3total = G3total + G3[i][0]
        G3average = G3total / 395
        print(session['session_username'])
        cur.execute(sql8, (session['session_username']))
        name = cur.fetchall()
        print(name[0][0])
        cur.execute(sql9, (name[0][0]))
        id = cur.fetchall()
        print(id[0][0])
        if id[0][0] == name[0][0]:
            return render_template('comparison.html',G1average=G1average,G2average=G2average,G3average=G3average,x=id[0][11],y=id[0][12],z=id[0][13])
            conn.commit()

    except:
        conn.rollback()
        return render_template('mainpage.html',x=2)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    f = open('LR_Model for more features and g1 g2', 'rb')  # 注意此处model是rb
    s = f.read()
    model = pickle.loads(s)
    conn = pymysql.connect(host='127.0.0.1', user='root', password='hushanxin.', db='RUNOOB', port=3306,
                           charset='utf8')
    cur = conn.cursor()
    sql10 = "select failures,Medu,higer,absences ,Fedu ,G1 ,G2  from project "
    sql11 = "select * from user_table where user_name = %s "
    sql12 = "select * from project where student_id = %s "
    try:
        cur.execute(sql10)
        features = cur.fetchall()
        print(features)
        df = pd.DataFrame(features, dtype=float)
        predictions = model.predict(df)
        print(predictions)
        cur.execute(sql11, (session['session_username']))
        name = cur.fetchall()
        cur.execute(sql12, (name[0][0]))
        id = cur.fetchall()
        x=id[0][0]
        print(x)
        if id[0][0] == name[0][0]:
            return render_template('prediction.html',score=predictions[x])
            conn.commit()
    except:
        conn.rollback()
        return render_template('mainpage.html', x=2)

@app.route('/overall', methods=['GET', 'POST'])
def overall():
    conn = pymysql.connect(host='127.0.0.1', user='root', password='hushanxin.', db='RUNOOB', port=3306,
                           charset='utf8')
    cur = conn.cursor()
    sql5 = "select G1 from project "
    sql6 = "select G2 from project "
    sql7 = "select G3 from project "
    try:
        cur.execute(sql5)
        G1 = cur.fetchall()
        G1_0_3=0
        G1_4_7=0
        G1_8_11=0
        G1_12_15=0
        G1_16_20=0
        G2_0_3 = 0
        G2_4_7 = 0
        G2_8_11 = 0
        G2_12_15 = 0
        G2_16_20 = 0
        G3_0_3 = 0
        G3_4_7 = 0
        G3_8_11 = 0
        G3_12_15 = 0
        G3_16_20 = 0
        for i in range(0, 395):
            if(0<=G1[i][0]<=3):
                G1_0_3=G1_0_3+1
            if (4<= G1[i][0] <= 7):
                G1_4_7 = G1_4_7 + 1
            if (8 <= G1[i][0] <= 11):
                G1_8_11 = G1_8_11 + 1
            if (12 <= G1[i][0] <= 15):
                G1_12_15 = G1_12_15 + 1
            if (16 <= G1[i][0] <= 20):
                G1_16_20 = G1_16_20 + 1

        cur.execute(sql6)
        G2 = cur.fetchall()
        for i in range(0, 395):
            if (0 <= G2[i][0] <= 3):
                G2_0_3 = G2_0_3 + 1
            if (4 <= G2[i][0] <= 7):
                G2_4_7 = G2_4_7 + 1
            if (8 <= G2[i][0] <= 11):
                G2_8_11 = G2_8_11 + 1
            if (12 <= G2[i][0] <= 15):
                G2_12_15 = G2_12_15 + 1
            if (16 <= G2[i][0] <= 20):
                G2_16_20 = G2_16_20 + 1
        cur.execute(sql7)
        G3 = cur.fetchall()
        for i in range(0, 395):
            if (0 <= G3[i][0] <= 3):
                G3_0_3 = G3_0_3 + 1
            if (4 <= G3[i][0] <= 7):
                G3_4_7 = G3_4_7 + 1
            if (8 <= G3[i][0] <= 11):
                G3_8_11 = G3_8_11 + 1
            if (12 <= G3[i][0] <= 15):
                G3_12_15 = G3_12_15 + 1
            if (16 <= G3[i][0] <= 20):
                G3_16_20 = G3_16_20 + 1
        print(G1_0_3,G1_4_7)
        return render_template('overall.html', G1_0_3=G1_0_3,G1_4_7=G1_4_7,G1_8_11=G1_8_11,G1_12_15=G1_12_15,G1_16_20=G1_16_20
                               ,G2_0_3=G2_0_3,G2_4_7=G2_4_7,G2_8_11=G2_8_11,G2_12_15=G2_12_15,G2_16_20=G2_16_20,
                               G3_0_3=G3_0_3,G3_4_7=G3_4_7,G3_8_11=G3_8_11,G3_12_15=G3_12_15,G3_16_20=G3_16_20)
        conn.commit()
    except:
        conn.rollback()
        return render_template('mainpage.html', x=1)

@app.route('/search', methods=['GET', 'POST'])
def search():
    return render_template('search.html',x=1)

@app.route('/Search', methods=['GET', 'POST'])
def Search():
    conn = pymysql.connect(host='127.0.0.1', user='root', password='hushanxin.', db='RUNOOB', port=3306,
                           charset='utf8')
    cur = conn.cursor()
    student_name = request.form.get('search')
    sql2 = "select student_id from students_tbl where username = %s "
    sql3 = "select * from project where student_id = %s "
    print(student_name)
    student_datalist=[]
    try:
        cur.execute(sql2, (student_name))
        student_id = cur.fetchall()
        print(student_id)
        cur.execute(sql3, (student_id[0][0]))
        student_data = cur.fetchall()
        print(student_data)
        for i in range(0,14):
            student_datalist.append(student_data[0][i])
        return render_template('search.html',student_data=student_datalist,x=2)
    except:
        conn.rollback()
        return render_template('login.html')

    conn.close()

if __name__ == '__main__':
    app.run()
