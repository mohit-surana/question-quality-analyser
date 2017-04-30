import glob, os

for root, dirs, files in os.walk('.'):
    i = 0
    for file in files:
        if file.endswith('.DS_Store'):
            path = os.path.join(root, file)

            print("Deleting: %s" % (path))

            try:
                if os.remove(path):
                    print("Unable to delete!")
                else:
                    print("Deleted", path)
                    i += 1
            except:
                print(path)
print("Files Deleted: %d" % (i))