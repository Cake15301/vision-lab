# ============================================================
# pipeline.py —— 零件检测仪 v1
# 干什么：扫一个文件夹的零件图，逐件量尺寸、判合格、画标注图、出一张结果表。
# 注释规矩（Day42 起）：只写"为什么"，不写"是什么"。
# 参数的含义、单位、取值依据统一放 参数依据.md —— 改参数先改那张表，再改这里的值。
# 跑法：D:/python/python.exe pipeline.py（在 v1 里）或 D:/python/python.exe v1/pipeline.py（在任何文件夹）
# ============================================================

from pathlib import Path   # 造"地址牌"
import cv2                 # OpenCV：整条流水线的九站全在它这儿
import numpy as np         # numpy：造形态学用的小方块
import csv                 # 标准库的"表格读写"工具（Day12 用它读过成绩表，今天用它写）

# IN_DIR：原料文件夹 —— 从"脚本自己的位置"出发往下走两步
# __file__ 是 Python 天生就有的变量：本脚本在硬盘上的完整地址（Day43 亲眼看过）
# .parent  退一层：从 ".../v1/pipeline.py" 退到 ".../v1" 这个文件夹
IN_DIR = Path(__file__).parent / "assets" / "in"

# ============ 配置区：九站的旋钮 ============
# 每个值后面只写"为什么是这个值"；含义、单位和实测出处见 参数依据.md。
# ⚠️ 这些值是给这台相机、这个视野定的，换设备/换视野必须重标：
#    编码器实验里把 PPT 表 2 的阈值照抄过来，68 根条纹全部误判 NG（100% 误报）。
TARGET_W = 1000    # 与 W06 全程同规格：缩到同一宽度，历史数字才能直接对账
MEDIAN_K = 5       # 5 是实测最优：3×3 救不回 Day33 的椒盐坏点，7 以上开始啃零件边缘
MORPH_K  = 3       # 只肯擦"比印章小"的白渣：5×5 起会把轮齿边缘咬缺（Day36 差异涨到 3184）
MIN_AREA = 60      # 灰尘最大 14.5、零件 186121：取两者中间，两头余量 4 倍 / 3000 倍

# ============ 图纸要求：标称值 ＋ 公差 ============
# 标称值不是抄书本，是拿标准件量出来的首件 —— 真实产线做"首件标定"也是这么定第一版
SPEC_AREA = 186121.0   # 标准件面积（平方像素）
SPEC_FILL = 0.7373     # 标准件占空比（面积 ÷ 外接矩形面积，无量纲）
# 公差两条腿一起卡：① 至少是实测波动的十倍（不误杀好件）② 不能大到放过真缺陷（不漏判）
TOL_AREA = 0.05        # 实测波动 0.17%，给 5% 是 29 倍；再放宽到 19% 以上，偏小 18.7% 的件就漏判
TOL_FILL = 0.02        # 实测波动 0.0012，给 0.02 是 16 倍；再紧到 0.001，稍模糊的好件就被误判 NG

# ============ 发货区：标注图存这里 ============
# 同样跟着脚本走：脚本在哪，它就找哪儿的 assets/out，换电脑、换文件夹都不用改代码
OUT_DIR = Path(__file__).parent / "assets" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def measure(path):
    """量一件：喂一张图的地址，拿回一本体检记录（字典）。"""

    # ---------- 站①② 读图 ＋ 灰度 ----------
    img = cv2.imread(str(path))          # imread 不认 Path 对象，必须转成字符串再喂
    ih, iw = img.shape[:2]               # shape 是（高, 宽），别跟 resize 的（宽, 高）搞反
    gray = cv2.cvtColor(
        cv2.resize(img, (TARGET_W, int(ih * TARGET_W / iw))),   # 先统一宽度再量：不统一，换个拍摄距离数字全变
        cv2.COLOR_BGR2GRAY)

    # ---------- 站③ 滤波 ----------
    den = cv2.medianBlur(gray, MEDIAN_K)   # 用中值不用均值：椒盐坏点是极端值，取中位数才不被它带偏

    # ---------- 站④ 阈值 ----------
    # 阈值给 0 无所谓：加了 OTSU，阈值由直方图自己算；手调反而容易只对某一张图好
    _, mask = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---------- 站⑤ 形态学 ----------
    stamp = np.ones((MORPH_K, MORPH_K), np.uint8)          # 印章多大＝"多大的白渣我肯认作噪声"
    # 用开不用闭：开只减不增，擦掉渣后把零件补回原样；闭会把渣和零件连成一片
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, stamp)

    # ---------- 站⑥ 找轮廓 ＋ 站⑦ 面积筛 ----------
    # 只要最外层：零件内部挖的洞是设计特征，不是第二个零件（v1 的已知盲区，v2 处理）
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)     # SIMPLE 只留拐点：形状不变、点数最少（Day37 833 vs 2339）
    kept = [c for c in cnts if cv2.contourArea(c) >= MIN_AREA]   # 60 以下一律当灰尘：真零件最小也有 15 万

    # ---------- 兜底：一张图里一个零件都没找到 ----------
    if len(kept) == 0:
        # 找不到零件也照规矩交货：面积给 0 让 judge 判 NG，图照样带回去让 annotate 能画
        # （调用方不用写一堆 if，一张坏图不会让整批停下）
        return {"面积": 0.0, "宽": 0, "高": 0, "占空比": 0.0,
                "x": 0, "y": 0, "图": gray}

    # ---------- 站⑧ 量形 ----------
    part = sorted(kept, key=cv2.contourArea, reverse=True)[0]   # 取最大的：真有残余灰尘，它一定最小
    # 用正外接不用最小外接：标注框要摆得正、让人一眼读得懂；量朝向才轮到最小外接（Day38）
    x, y, w, h = cv2.boundingRect(part)
    # 用 contourArea 不用 w*h：前者量零件本体，后者量外框；占空比正好是这两者之比
    area = cv2.contourArea(part)

    # ---------- 装进箱子递回去 ----------
    # x、y、图 三样必须带回去：annotate 画框要用，而且必须是同一张图，否则框会错位
    # 占空比 = 轮廓面积 ÷ 外接矩形面积（怎么读：接近 1 表示方方正正占满框）
    return {"面积": area, "宽": w, "高": h, "占空比": area / (w * h),
            "x": x, "y": y, "图": gray}

def judge(m):
    """判一件：喂一本测量记录（m），拿回一张判决书（也是字典）。"""

    # 先判"量到了没有"：面积 0 说明没找到零件，后面算出的偏差都没有意义
    if m["面积"] == 0:
        return {"结论": "NG", "原因": "没找到零件"}

    # 用相对偏差不用绝对差：换成别的零件尺寸，5% 这个门槛还能继续用
    devi_area = round((m["面积"] - SPEC_AREA) / SPEC_AREA, 4)

    # 先砍到 4 位再相减：不砍，0.7372763… 这种尾巴会算出 -0.00002 的假偏差
    devi_fill = round(round(m["占空比"], 4) - SPEC_FILL, 4)

    # 先面积后形状：面积都不合格，"形状合格"这句话就没有意义，第一道不过就此打住
    # 上下分开写：将来要不对称公差（比如只许偏大不许偏小），只改这两个 TOL 就够
    if devi_area < -TOL_AREA or devi_area > TOL_AREA:
        return {"结论": "NG", "原因": f"面积偏差 {round(devi_area * 100, 1)}%"}

    # 形状闸管面积闸管不到的事：面积对但缺角、多料、变形
    # （整体缩小的件形状不变，所以只有面积闸拦得住它 —— 两道闸各管一件事，缺一不可）
    if devi_fill < -TOL_FILL or devi_fill > TOL_FILL:
        return {"结论": "NG", "原因": f"占空比偏差 {devi_fill}"}

    # 两道闸都过才叫合格：宁可让"临界合格"流到人工复检，也不在这里放宽
    return {"结论": "OK", "原因": "尺寸、形状都在公差内"}


def annotate(m, r):
    """画一张判决图：喂测量记录 m 和判决书 r，拿回一张画好框和字的彩色图。

    只管"画"：存不存、存到哪儿，交给外面决定。
    """

    # 灰度只有 1 层，写不进 3 个数一组的颜色，先补成 3 层（Day39 的 to_bgr 就是干这个）
    view = cv2.cvtColor(m["图"], cv2.COLOR_GRAY2BGR)

    # 颜色顺序是 (蓝, 绿, 红)，跟日常说法反着；红留给不合格，因为产线上"坏的"必须最扎眼
    if r["结论"] == "OK":
        color = (0, 200, 0)
    else:
        color = (0, 0, 255)

    x, y, w, h = m["x"], m["y"], m["宽"], m["高"]

    # 右下角＝左上角＋宽高，不重新量：量的和画的必须出自同一次处理
    cv2.rectangle(view, (x, y), (x + w, y + h), color, 3)

    # putText 的位置是文字左下角：写 y-12 让字浮在框上方；零件贴画面顶边时改 y+30 塞进框里
    cv2.putText(view, r["结论"] + "  " + r["原因"], (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return view

def main():
    """跑一批：扫 IN_DIR 里所有 png，逐件量、判、画、存，最后出汇总和结果表。"""
    # ==================== 主循环：把 assets/in 里每一张图都判一遍、画一张、存一张 ====================
    # 先把"图纸"念一遍 —— 让人知道这份结论是拿什么标准判的
    print(f"图纸：面积 {SPEC_AREA} 平方像素（公差 ±5%）；占空比 {SPEC_FILL}（公差 ±0.02）")
    print()

    # rows：整张表。先放表头那一行，后面每判完一件就往里加一行
    rows = [["文件名", "面积", "宽", "高", "占空比", "结论", "原因"]]

    files = sorted(IN_DIR.glob("*.png"))   # 名单：文件夹里所有 png，按文件名排好序（顺序固定才好对账）
    ok_n = 0                               # 计数器：合格几件，从 0 开始

    for p in files:                        # p：名单里当前这一张的地址牌
        m = measure(p)                     # 站①~⑧：量这一件，拿回箱子（里面还带着图）
        r = judge(m)                       # 站⑨  ：拿箱子对图纸，拿回判决书
        view = annotate(m, r)              # 站⑩  ：把判决书画到图上

        # 存图：文件名 = anno_ + 原名 —— 只加前缀不改名，一眼看出画的是哪张原图
        save = OUT_DIR / ("anno_" + p.name)
        done = cv2.imwrite(str(save), view)     # 成功回 True，失败回 False
        if done is False:                       # ⚠️ 存盘失败不会抛异常，只会悄悄回 False —— 必须自己查
            print("⚠️ 存盘失败：", save)

        if r["结论"] == "OK":
            ok_n += 1                       # += 1 就是"自己加一"（Day8 学过）

        # 把这一件的体检数据攒成一行，追加到整张表的末尾
        rows.append([p.name, round(m["面积"], 1), m["宽"], m["高"],
                    round(m["占空比"], 4), r["结论"], r["原因"]])

        print(p.name, "| 面积", round(m["面积"], 1), "| 框", m["宽"], "x", m["高"],
            "| 占空比", round(m["占空比"], 4), "|", r["结论"], "-", r["原因"])

    # ---------- 写结果表 ----------
    TABLE = OUT_DIR / "结果表.csv"       # 表存哪儿（和标注图放一起）
    # newline=""        ：不让 Python 自己再补换行；不写这句，Windows 上每两行中间会多出一个空行
    # encoding="utf-8-sig"：文件开头加一个"我这是 UTF-8"的标记；不加，Excel 打开中文表头是乱码
    with open(TABLE, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)    # writerows = 一次把整张表（含表头）全写进去
    print("结果表 ->", TABLE)

    print()
    # 汇总：NG 的件数不用单独数，总数减掉 OK 的就是
    print(f"共 {len(files)} 件：OK {ok_n} 件，NG {len(files) - ok_n} 件")
    print(f"标注图 -> {OUT_DIR}")

if __name__ == "__main__":
    main()