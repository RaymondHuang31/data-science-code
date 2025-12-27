import os


def make(path):
    lists = []
    kinds = os.listdir(path)
    for i in range(len(kinds)):
        class_path = os.path.join(path, kinds[i])
        for j in os.listdir(class_path):
            lists.append("%s %d" % (os.path.join(class_path, j), i))
    file = open("%s.txt" % path, 'w')
    file.write('\n'.join(lists))
    file.close()

if __name__ == "__main__":
    make('train')
    make('val')
    make('test')