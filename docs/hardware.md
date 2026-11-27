# 实机巡检接入与验收

此页覆盖真实 UR5e 与 USB 相机的巡检接入，以及独立实机验收。三螺丝装配、腕部视觉定位和电批异常恢复的现有验证范围为 MuJoCo。命令从仓库根目录执行，环境安装见 [README](../README.md)。

```powershell
$inspectionPython = '.\.venv\Scripts\python.exe'
```

## 接入真实 UR5e 与相机

实机驱动使用 PolyScope 5 的 URScript 和 Windows USB UVC 相机。PC 发出拍摄位编号，控制器执行示教路径并确认到位和停止后，PC 取新帧。实机使用示教轨迹和实物照片，仿真的关节位及模板不参与实机运行。当前程序已测试 PC 端协议，URScript 仍需在目标控制器验证。

1. 复制 `station/example.json` 为 `station/local.json`。填入 Windows 网卡和 UR5e 的地址、相机编号和实际图像分辨率。`home` 是控制器读出的六关节弧度值。`routes` 的每个视角是从公共待机位出发的一组关节点，最后一个是拍摄位，前面可加过渡点。每次换拍摄位先沿原路线退回待机位，再走目标路线。现场先确认工具和负载参数，低速逐段验证六条去回路径与夹具间隙。
2. 从配置生成 `station/local.script`。在 PolyScope 5 程序中添加 Script 节点，选 File 并加载脚本。把机械臂置于示教待机位，现场启动该程序。站点程序在待机位等待 Windows 开始一轮，结束后返回待机位等待下一轮。若通信中断，站点程序停止派发新动作。恢复时按现场操作流程确认当前姿态，回到示教待机位再启动。控制器使用进站握手验证当前示教配置的指纹。修改轨迹后重新生成脚本并更新控制器上的版本。[PolyScope Script 节点](https://www.universal-robots.com/manuals/EN/HTML/SW5_26/Content/prod-usr-man/software/PolyScope/content/AdvProgNodes/commandtab_script_en.htm) · [URScript TCP/IP 接口](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/tcp-ip-client.html)
3. 用 `collect` 保存实物样图。每张照片需来自对应螺丝位、视角和最终相机分辨率。准备完好和缺件样件，标签由操作者按实际缺件位填写。复制 `station/profile.example.json` 为 `station/profile.local.json`，将参考 PNG 放进 `station/references/`，按相同的 `site` 和 `angle` 填入 `present`、`missing` 的文件路径。每组允许多张图，路径相对于配置文件。将来自其他工件或其他轮次的图片留作独立验证。
4. 校准六个视角的 `roi`，格式为 `[左, 上, 宽, 高]`，单位为像素。ROI 要覆盖螺丝头和孔缘。`shift_px` 是允许的微小位置偏移。用独立实物样图检验 `min_sharpness`（拉普拉斯方差）、`max_glare` 与 `max_dark`（极亮、极暗区域占比），以及 `min_score` 和 `min_margin`（最小匹配分、两类分差）。`samples.csv` 的列为 `site,angle,expected,image`，`expected` 填 `present`、`missing` 或 `review`。反光、模糊、遮挡和位置偏移的实物样图有助于验证待复检门限。
5. 离线验证通过后，用实机配置打开操作台。装夹工件后，每点一次 **开始质检**，执行并保存一轮。

```powershell
& $inspectionPython -m inspection build-station --station station/local.json --output station/local.script
& $inspectionPython -m inspection collect --station station/local.json '--missing='
& $inspectionPython -m inspection collect --station station/local.json --missing 'S1,S2,S3'
& $inspectionPython -m inspection verify-samples --profile station/profile.local.json --manifest real_holdout/samples.csv --output outputs/hardware-samples.json
& $inspectionPython -m inspection gui --station station/local.json --profile station/profile.local.json
```

## 实机 20 轮验收

用完好件和缺件分别跑若干轮，共 20 轮。每轮由操作者先独立记录实际缺件位置，建议 10 轮完好、10 轮缺件，并覆盖 S1、S2、S3。把每轮的 `report.json` 路径填入 CSV，每个报告只填一次。`missing` 留空表示完好，多处缺件用 `|` 分隔。示例两行用于说明格式，正式清单须包含 20 轮。

```csv
missing,report
,outputs/runs/<完好轮次编号>/report.json
S1|S3,outputs/runs/<缺件轮次编号>/report.json
```

```powershell
& $inspectionPython -m inspection evaluate --manifest trials.csv --source hardware --output outputs/hardware-acceptance.json
```

验收程序要求整件命中至少 18/20 且缺件判为合格次数为 0。待复检计入未判对。统计也记录各螺丝位判对数量，实机验收只收实机来源的报告。当前验证数据均来自 MuJoCo，实物样图标定和实机统计尚待现场执行。
