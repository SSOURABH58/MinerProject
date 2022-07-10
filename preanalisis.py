

# data = "color"
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Image

data = "color"

category_names = sorted(os.listdir(data))
nb_categories = len(category_names) # number of category --
img_pr_cat = []
for category in category_names:
    folder = data + '/' + category
    img_pr_cat.append(len(os.listdir(folder)))
print(category_names)
sns.barplot(y=category_names, x=img_pr_cat).set_title("Number of training images per category:")

fig = plt.figure(figsize=(10, 7))


rows = math.sqrt(nb_categories)
columns = rows+1

count=-1
for subdir, dirs, files in os.walk(data):
    count+=1
    for i,file in enumerate(files):
        img_file = subdir + '/' + file
        image = mpimg.imread(img_file)
        img = image.resize((500, 500))
        # plt.figure()
        print(count)
        fig.add_subplot(rows, columns, count)
        plt.axis('off')
        # plt.title(subdir)
        plt.imshow(img)
        break

plt.show()