# ===== 段 1：建好"发货区"，数清"来货" =====
# 这一段干完，你手里有：一个建好的输出文件夹 out/，和一张 7 张图的名单 files

from pathlib import Path   # Path：造"地址牌"的工具（Day15 学过）
import cv2                 # OpenCV：后面几段读图、缩放、灰度、存图都靠它

# IN_DIR ：原料文件夹（7 张原始样本）的地址牌
# OUT_DIR：成品文件夹的地址牌（现在可能还不存在）
IN_DIR  = Path("D:/AI/zcode files/vision-lab/practice/day31_assets/in")
OUT_DIR = Path("D:/AI/zcode files/vision-lab/practice/day31_assets/out")

# 在存文件之前，先确保"发货区"这个文件夹真的躺在磁盘上
# exist_ok=True ：已经有了也不算错（工具每天要跑很多遍，不能报"文件夹已存在"）
# parents=True  ：上级文件夹缺了就一起建出来
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("输出文件夹存在吗 =", OUT_DIR.exists())   # .exists() 回 True 或 False

# glob("*.jpg")：让 IN_DIR 里所有 .jpg 文件排队过来
#                （它给的是"传送带"，不是列表，所以不能直接数长度）
# sorted(...) ：把传送带装成"名单"，并按文件名排好序
#                （顺序固定，你才能一行一行对着输出核对）
files = sorted(IN_DIR.glob("*.jpg"))
print(f"输入文件夹里有 {len(files)} 张图")

# 逐个打印名单里的名字：p 是当前这一项的地址牌
# p.name：只要文件名（带后缀），丢掉前面的长路径
for p in files:
    print(p.name)

# ===== 段 2：拿第一张图，把整条处理链走一遍 =====

p = files[0]
img = cv2.imread(p)
print("文件名 =", p.name, " shape =", img.shape)

h = img.shape[0]
w = img.shape[1]
print("原始 高 =", h, " 宽 =", w)

TARGET_W = 800
scale = TARGET_W / w
new_h = int(h * scale)
print("缩放比例 =", round(scale, 4))
print("目标尺寸 = 宽", TARGET_W, "高", new_h)

small = cv2.resize(img, (TARGET_W, new_h))      #shape 的顺序是高、宽、通道，resize 要的顺序是宽、高!
print("缩放后 shape =", small.shape)

gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
print("灰度后 shape =", gray.shape)

final = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
print("调完对比度 shape =", final.shape, " 平均亮度 =", round(final.mean(), 2))

# ===== 段 3：批量处理——同一套处理链，7 张一起过并存盘 =====
# 这一段干完，out/ 里会出现 7 张同名的 800x533 灰度图

for p in files:                       # 名单里的图轮流进来，p 依次换成每一张的地址牌
    img = cv2.imread(p)               # 读原图（每张约 57 MB 内存）
    h = img.shape[0]                  # 高
    w = img.shape[1]                  # 宽

    new_h = int(h * TARGET_W / w)     # 按宽度等比算出新高度（TARGET_W 段 2 已定为 800）
    small = cv2.resize(img, (TARGET_W, new_h))            # 缩到（宽, 高）
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)        # 转灰度
    final = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)  # 拉对比度

    # 输出地址 = 输出文件夹 拼上 原文件名 —— 只换文件夹、不换名，方便追溯原图
    save_path = OUT_DIR / p.name
    ok = cv2.imwrite(save_path, final)   # 存盘：成功回 True，失败回 False
    print(p.name, "->", ok, final.shape)

# ===== 段 4：体检——每张图算平均亮度，分"太暗 / 正常 / 过曝"三档 =====
# 这一段干完，report 袋子里装着 7 行文字（每张一行的体检结论）

TOO_DARK   = 55     # 平均亮度低于这个数就算"太暗"（按这批图的光照定的，不是通用值）
TOO_BRIGHT = 200    # 平均亮度高于这个数就算"过曝"

report = []          # 空袋子：每张图体检完，把结论行塞进来

for p in files:
    img = cv2.imread(p)               # 读原图
    h = img.shape[0]
    w = img.shape[1]

    new_h = int(h * TARGET_W / w)
    small = cv2.resize(img, (TARGET_W, new_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)   # 体检在灰度图上量

    mean_v = gray.mean()   # 体检指标：灰度图平均亮度（0 全黑，255 全白）

    # 三档判定：从上往下问，哪个先成立就停在哪一档
    if mean_v < TOO_DARK:
        tag = "太暗"
    elif mean_v > TOO_BRIGHT:
        tag = "过曝"
    else:
        tag = "正常"

    # 拼成一行人话：文件名 ＋ 亮度（只留两位小数）＋ 结论
    line = f"{p.name}  亮度 {round(mean_v, 2)}  {tag}"
    print(line)
    report.append(line)                # 把这一行记进袋子

print()
print("清单条数 =", len(report))

# ===== 段 5：把体检表写进 txt 文件 =====
# 这一段干完，out/体检报告.txt 里会有 7 行，和段 4 打印的一模一样

report_path = OUT_DIR / "体检报告.txt"   # 报告文件的地址牌

# with open(...) ：打开文件准备写，"w" 是写模式（没有就新建，有就覆盖）
# encoding="utf-8"：中文按 utf-8 存，不写这句，报告里的中文会变乱码
with open(report_path, "w", encoding="utf-8") as f:
    for line in report:          # 袋子里的每一行体检结论
        f.write(line + "\n")     # 写进去，再补一个换行符 "\n"（不然全部挤成一行）
print("体检报告 ->", report_path)