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
        print(f"\rتحميل: {pct:5.1f}% ({done/1e6:.0f} / {total_size/1e6:.0f} MB)", end="")

# اعرف الحجم الحقيقي من السيرفر
with urllib.request.urlopen(url) as r:
    expected = int(r.headers["Content-Length"])

if os.path.exists(archive) and os.path.getsize(archive) != expected:
    print("الملف ناقص، هنمسحه ونحمّل من الأول")
    os.remove(archive)

if not os.path.exists(archive):
    urllib.request.urlretrieve(url, archive, progress)
    print()

print("حجم الملف:", os.path.getsize(archive), "من", expected)

print("فك الضغط...")
with tarfile.open(archive) as t:
    t.extractall(out_dir)
print("تمام:", os.path.abspath(out_dir))