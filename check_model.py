from modelscope.utils.constant import DownloadMode
from modelscope.hub.snapshot_download import snapshot_download

# 再次调用下载命令，并将 mode 设置为先校验
# 如果文件完整，它会秒级扫描并提示 "File exists"
path = snapshot_download(
    'qwen/Qwen3.6-27B', 
    cache_dir='/root/rivermind-data/modelscope_models',
    download_mode=DownloadMode.FORCE_REDOWNLOAD # 这会强制触发校验逻辑
)