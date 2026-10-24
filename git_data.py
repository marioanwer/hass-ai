import os
import tarfile
import urllib.request

url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
archive = "speech_commands_v0.02.tar.gz"
out_dir = "speech_commands"

def progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(done / total_size * 100, 100)

with urllib.request.urlopen(url) as r:
    expected = int(r.headers["Content-Length"])

if os.path.exists(archive) and os.path.getsize(archive) != expected:
    os.remove(archive)

if not os.path.exists(archive):
    urllib.request.urlretrieve(url, archive, progress)
    print()

with tarfile.open(archive) as t:
    t.extractall(out_dir)
