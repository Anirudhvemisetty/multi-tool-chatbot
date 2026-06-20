from youtube_transcript_api import (
    YouTubeTranscriptApi
)

api = YouTubeTranscriptApi()

transcript = api.fetch(
    "jNQXAC9IVRw"
)

print(transcript[:5])