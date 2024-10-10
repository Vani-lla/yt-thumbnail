from dataclasses import dataclass
import numpy as np

from pytubefix import Channel, YouTube
from requests import get
from json import dump
from concurrent import futures


@dataclass
class Thumbnail:
    _id: str
    _url: str
    views: str
    succesfull: bool

    def download_thumbnail(self) -> None:
        response = get(self._url)

        path = f"data/thumbnails/succesfull/{
            self._id}.jpg" if self.succesfull else f"data/thumbnails/failure/{self._id}.jpg"

        with open(path, "wb") as thumbnail:
            thumbnail.write(response.content)


def get_information_from_video(video: YouTube) -> tuple[int, str, str]:
    return video.views, Thumbnail(video.video_id, video.thumbnail_url, video.views, False)


# Open channels from a file
with open("data/channels.txt") as file:
    channels = map(lambda line: Channel(line.rstrip()), file.readlines())

with futures.ThreadPoolExecutor(max_workers=16) as e:
    for videos in map(lambda c: c.videos, channels):
        views, thumbnails = zip(*e.map(get_information_from_video, videos))

        # Calculate median views to derermine if video was a success
        channel_median = np.median(views)
        for tn in thumbnails:
            tn.succesfull = tn.views > channel_median

        list(e.map(lambda tn: tn.download_thumbnail(), thumbnails))
