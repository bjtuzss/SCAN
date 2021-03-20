from flask import Flask, request, jsonify
import base64
import os
import SCAN_master.test_SCRN as scan

app = Flask(__name__)
count = 0


# 清空图片文件夹
def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/handle', methods=['get', 'post'])
def handle():
    del_files("SCAN_master/resources/")
    del_files("SCAN_master/result/bestnet-Finally2/")
    print('-------------')
    requestdata = request.json
    img = requestdata.get('base64')
    arry = img.split(",")
    msg = arry[1]
    print(msg) # 图片的base64内容
    style = arry[0].split("/")
    style_finall = style[1].split(";")
    print(style_finall[0])  # 图片形式

    # 保存base64图片
    imagedata = base64.b64decode(msg.encode())
    filename = 'SCAN_master/resources/1.' + style_finall[0]
    fh = open(filename, "wb")
    fh.write(imagedata)
    fh.close()
    scan.handle_scan()

    # 图片转base64
    openname = 'SCAN_master/result/bestnet-Finally2/1binary.' + style_finall[0]
    with open(openname, 'rb') as f:
        result_encode = base64.b64encode(f.read())
        result_edecode = result_encode.decode()
        print(result_edecode)
        result = arry[0] + ',' + result_edecode
        print(result)


    return jsonify({
        'base64': result
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
