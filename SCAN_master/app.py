from flask import Flask, request, jsonify
import base64
import os
from SCAN_master.test_SCRN_F import Scan_Master

app = Flask(__name__)
count = 0
# 预处理
scan = Scan_Master()


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


# 预处理
del_files("SCAN_master/resources/")
del_files("SCAN_master/result/")


# 图片转base64
def to_base64(openname, style):
    with open(openname, 'rb') as f:
        result_encode = base64.b64encode(f.read())
        result_edecode = result_encode.decode()
        result = style + ',' + result_edecode

    return result


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/handle', methods=['get', 'post'])
def handle():
    del_files("SCAN_master/resources/")
    del_files("SCAN_master/result/")
    requestdata = request.json
    img = requestdata.get('base64')
    arry = img.split(",")
    # 存储base64的具体内容
    msg = arry[1]
    # 分割出单纯的 .png/.jpg
    style = arry[0].split("/")
    style_finall = style[1].split(";")

    # 保存base64图片
    imagedata = base64.b64decode(msg.encode())
    filename = 'SCAN_master/resources/1.' + style_finall[0]
    fh = open(filename, "wb")
    fh.write(imagedata)
    fh.close()
    # 调用算法，获取结果并直接存储在result里面
    scan.handle_scan()
    # 图片转base64
    openname = 'SCAN_master/result/bestnet-Finally2/1.' + style_finall[0]
    re = to_base64(openname, arry[0])

    return jsonify({
        'base64': re
    })


@app.route('/getMsg', methods=['get', 'post'])
def getMessages():
    # 首先判断文件是否为空
    f_path = "SCAN_master/result/bestnet-Finally2/"
    files_list = os.listdir(f_path)
    if len(files_list) > 0:
        # 获得信息
        data = scan.getMsg()
        return jsonify({
            'message': data
        })
    else:
        return jsonify({
            'message': 'wrong'
        })


# 获取效果图
@app.route('/getPics', methods=['get', 'post'])
def getPictures():
    sty = 'data:image/png;base64'
    pic_path1 = "SCAN_master/result/bestnet-Finally2GGRNet_fm_curves.png"
    pic_path2 = "SCAN_master/result/bestnet-Finally2GGRNet_pr_curves.png"
    s = [to_base64(pic_path1, sty),to_base64(pic_path2, sty)]
    return jsonify({
            'pic1': s[0],
            'pic2': s[1]
        })


# 获取gt效果图
@app.route('/getGt', methods=['get', 'post'])
def getPicture_gt():
    requestdata = request.json
    name = requestdata.get('name')
    sty = 'data:image/png;base64'
    pic_gt = "SCAN_master/gt/1.png"
    print(to_base64(pic_gt, sty))
    return jsonify({
            'pic': to_base64(pic_gt, sty)
        })


# 获取gt效果图
@app.route('/getCom', methods=['get', 'post'])
def getPicture_com():
    sty = 'data:image/png;base64'
    pic_com = "SCAN_master/comparison/comparison.png"
    print(to_base64(pic_com, sty))
    return jsonify({
            'pic': to_base64(pic_com, sty)
        })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
