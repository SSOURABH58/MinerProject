import os
import shutil

data = "color"
test_dest = "testing"

category_names = sorted(os.listdir(data))

final_look = {}

for i,catogry in enumerate(category_names):
    print(catogry)
    final_look[catogry]=[]
    images = sorted(os.listdir(data+"/"+catogry))
    os.mkdir(test_dest+"/"+catogry)
    for j in range(3):
        source = data+"/"+catogry+"/"+images[j]
        destination = test_dest + "/" + catogry + "/" + images[j]
        print(source," -->to--> ",destination)
        # shutil.copy2(source, destination)
        os.replace(source, destination)
        final_look[catogry].append(destination)

print(final_look)
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
