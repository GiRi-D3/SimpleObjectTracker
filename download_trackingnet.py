from huggingface_hub import login
login(token="<<---generate and use your own token--->>") # https://huggingface.co/settings/tokens -> Create a new token

from TrackingNet.Downloader import TrackingNetDownloader
downloader = TrackingNetDownloader(LocalDirectory="TrackingNet")

## change the split no. to either "TEST" or "TRAIN_*" (replace * with a number between 0 to 11)
downloader.downloadSplit("TRAIN_0")
