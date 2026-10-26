# ============================================================
# pipeline.py —— 我的零件检测仪 v1（W07 第 1 天）
# 干的事：喂它一张零件图，它量出尺寸，再判断这一件合不合格
# 九个站：读图 → 灰度 → 滤波 → 阈值 → 形态学 → 找轮廓 → 面积筛 → 量形 → 判合格
# 跑法（在 v1 文件夹里）：D:/python/python.exe pipeline.py
# ⚠️ 跑之前要先跑一次 make_samples.py 造样件，否则 assets/in 里没有图
# ============================================================

from pathlib import Path   # 造"地址牌"
import cv2                 # OpenCV：整条流水线的九站全在它这儿
import numpy as np         # numpy：造形态学用的小方块

# IN_DIR：原料文件夹（三件样件在这里）
IN_DIR = Path("D:/AI/zcode files/vision-lab/practice/v1/assets/in")

# ============ 配置区：九站的旋钮（为什么这么设见 day39_assets/参数说明表.md）============
TARGET_W = 1000    # 站①：统一缩到 1000 像素宽（和 W05/W06 同规格，数字才能对账）
MEDIAN_K = 5       # 站③：中值滤波窗口边长（像素，必须奇数）—— 专治椒盐坏点
MORPH_K  = 3       # 站⑤：形态学印章边长（像素）—— 只擦比 3×3 小的白渣
MIN_AREA = 60      # 站⑦：面积闸门（平方像素）—— 比它小的当灰尘扔掉

# ============ 图纸要求：标称值 ＋ 公差（v1 的"标准"）============
# 标称值不是抄书本，是段 3 拿标准件量出来的 —— 真实产线做"首件标定"也是这么定第一版。
SPEC_AREA = 186121.0   # 标准件面积（平方像素）—— 段 3 屏幕上的"面积"
SPEC_FILL = 0.7373     # 标准件占空比（无量纲）—— 段 3 屏幕上的"占空比"
# 公差定法：测量重复性的 10 倍以上（面积重复性 0.0008% → 给 ±5%）
TOL_AREA = 0.05        # 面积允许偏差 ±5%（0.05 就是 5%，写小数方便直接算）
TOL_FILL = 0.02        # 占空比允许偏差 ±0.02（直接相减的绝对差，不是百分比）
# ===============================================================


def measure(path):
    """量一件：喂一张图的地址，拿回一本体检记录（字典）。"""

    # ---------- 站①② 读图 ＋ 灰度 ----------
    img = cv2.imread(str(path))          # 读图；路径要 str() 包一层
    ih, iw = img.shape[:2]               # 高 ih、宽 iw（shape 先高后宽）
    gray = cv2.cvtColor(
        cv2.resize(img, (TARGET_W, int(ih * TARGET_W / iw))),   # 先缩到 1000 宽
        cv2.COLOR_BGR2GRAY)                                     # 再压成灰度

    # ---------- 站③ 滤波 ----------
    den = cv2.medianBlur(gray, MEDIAN_K)   # 窗口里取中位数：孤立的黑白坏点被邻居"投票"顶掉

    # ---------- 站④ 阈值（Otsu 自动定）----------
    # 第 2 个参数给 0 没关系：加了 THRESH_OTSU，阈值由它自己从直方图算
    _, mask = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---------- 站⑤ 形态学（开运算）----------
    stamp = np.ones((MORPH_K, MORPH_K), np.uint8)          # 3×3 方块印章
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, stamp)  # 开运算＝先腐蚀再膨胀：擦掉小白渣

    # ---------- 站⑥ 找轮廓 ＋ 站⑦ 面积筛 ----------
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,      # EXTERNAL：只要最外层（数零件用）
                               cv2.CHAIN_APPROX_SIMPLE)       # SIMPLE：只记拐点，点少算得快
    kept = [c for c in cnts if cv2.contourArea(c) >= MIN_AREA]   # 面积闸门：小的当灰尘扔掉

    # ---------- 兜底：一张图里一个零件都没找到 ----------
    if len(kept) == 0:
        # 还是得递回一个"箱子"，只是里面全是 0 —— 调用方拿到 0 就知道"没找到"
        return {"面积": 0.0, "宽": 0, "高": 0, "占空比": 0.0}

    # ---------- 站⑧ 量形 ----------
    part = sorted(kept, key=cv2.contourArea, reverse=True)[0]   # 面积最大的那个当目标
    x, y, w, h = cv2.boundingRect(part)                        # 正外接矩形：左上角(x,y)＋宽w＋高h
    area = cv2.contourArea(part)                               # 轮廓面积（平方像素）

    # ---------- 装进箱子递回去 ----------
    # 占空比 = 轮廓面积 ÷ 外接矩形面积（无量纲）—— 零件"占满外框的程度"
    return {"面积": area, "宽": w, "高": h, "占空比": area / (w * h)}

def judge(m):
    """判一件：喂一本测量记录（m），拿回一张判决书（也是字典）。"""

    # 兜底：measure 说"没找到零件"，那就直接判 NG，别往下算（0 除以 SPEC 也能算，但没意义）
    if m["面积"] == 0:
        return {"结论": "NG", "原因": "没找到零件"}

    # 面积偏差：小数形式，-0.187 就是"小了 18.7%"。除以 SPEC 是为了变成"相对偏差"
    devi_area = round((m["面积"] - SPEC_AREA) / SPEC_AREA, 4)

    # 占空比偏差：先把测到的占空比砍到 4 位小数，再跟标称值相减
    # （先砍四位是为了消掉 0.7372763… 这种尾巴，不然标准件会算出一个 -0.00002 的假偏差）
    devi_fill = round(round(m["占空比"], 4) - SPEC_FILL, 4)

    # 第一道闸：面积超差 → NG（零件大了、小了，或者根本没装上）
    # "小于下公差" 或 "大于上公差" 分开写：将来要改成不对称公差，只改这两个 TOL 就行
    if devi_area < -TOL_AREA or devi_area > TOL_AREA:
        return {"结论": "NG", "原因": f"面积偏差 {round(devi_area * 100, 1)}%"}

    # 第二道闸：占空比超差 → NG（面积对、但形状不对：缺角、多料、变形）
    if devi_fill < -TOL_FILL or devi_fill > TOL_FILL:
        return {"结论": "NG", "原因": f"占空比偏差 {devi_fill}"}

    # 两道闸都过了，才判 OK
    return {"结论": "OK", "原因": "尺寸、形状都在公差内"}

# ==================== 主循环：把 assets/in 里每一张图都判一遍 ====================
# 先把"图纸"念一遍 —— 让人知道这份结论是拿什么标准判的
print(f"图纸：面积 {SPEC_AREA} 平方像素（公差 ±5%）；占空比 {SPEC_FILL}（公差 ±0.02）")
print()

# sorted(目录.glob("*.png"))：把文件夹里所有 png 排好序排成名单（Day31 学过）
for p in sorted(IN_DIR.glob("*.png")):
    m = measure(p)      # 站①~⑧：量这一件，拿回箱子
    r = judge(m)        # 站⑨  ：拿箱子对图纸，拿回判决书
    print(p.name, "| 面积", round(m["面积"], 1), "| 框", m["宽"], "x", m["高"],
          "| 占空比", round(m["占空比"], 4), "|", r["结论"], "-", r["原因"])

