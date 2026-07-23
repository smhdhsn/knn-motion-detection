import yt_dlp


def get_stream_url(youtube_url, max_height=720):
    ydl_opts = {
        "format": f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(youtube_url, download=False)["url"]
