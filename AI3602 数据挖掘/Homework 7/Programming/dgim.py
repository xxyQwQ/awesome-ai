import os


file_dir = os.path.dirname(__file__)
data_path = os.path.join(file_dir, "stream_data_dgim.txt")
edge_path1 = os.path.join(file_dir, "edge1.txt")
edge_path2 = os.path.join(file_dir, "edge2.txt")


class DGIM:
    def __init__(self, filepath, windowsize, maxtime=None):
        self.fileHandler = open(filepath, 'r')
        self.windowSize = windowsize
        self.buckets = []
        self.timeMod = maxtime if maxtime else windowsize << 2
        self.timestamp = 0

    def update(self, x):
        ### TODO
        if len(self.buckets) > 0 and (self.timestamp - self.windowSize + self.timeMod) % self.timeMod == self.buckets[0][0]:
            self.buckets.pop(0)
        if x == "0":
            return
        self.buckets.append([self.timestamp, 1])
        for i in range(len(self.buckets) - 1, 1, -1):
            if self.buckets[i - 2][1] == self.buckets[i][1]:
                self.buckets[i - 2][0] = self.buckets[i - 1][0]
                self.buckets[i - 2][1] += self.buckets[i - 1][1]
                self.buckets.pop(i - 1)
        ### end of TODO

    def run(self):
        f = self.fileHandler
        x = f.read(2).strip()
        while x:
            self.update(x)
            self.timestamp = (self.timestamp + 1) % self.timeMod
            x = f.read(2).strip()

    def count(self, k=None):
        if k is None:
            k = self.windowSize
        result = 0
        ### TODO
        if len(self.buckets) == 0:
            return 0
        i = len(self.buckets) - 1
        while i >= 0 and (self.timestamp - k + self.timeMod) % self.timeMod < self.buckets[i][0]:
            result += self.buckets[i][1]
            i -= 1
        result -= self.buckets[i + 1][1] // 2
        return result
        ### end of TODO


if __name__ == "__main__":
    dgim = DGIM(filepath=data_path, windowsize=1000)
    dgim.run()
    print("the number of 1-bits in current windows:")
    print(f"    {dgim.count()}")
    print("the number of 1-bits in the last 500 and 200 bits of the stream:")
    print(f"    {dgim.count(k=500)}")
    print(f"    {dgim.count(k=200)}")

    print("edge cases:")
    dgim1 = DGIM(filepath=edge_path1, windowsize=1000)
    dgim1.run()
    print(f"    {dgim1.count()}")
    dgim2 = DGIM(filepath=edge_path2, windowsize=1000)
    dgim2.run()
    print(f"    {dgim2.count()}")
