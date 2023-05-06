from ECMIFrameWork import ECMIFrameWork
from util.config import ModelConf


def runModel(num):
    path = f"./config/ECMI{num}.conf"
    print(path)
    conf = ModelConf(path)

    # except KeyError:
    #     print('wrong num!')
    #     exit(-1)
    recSys = ECMIFrameWork(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))


if __name__ == '__main__':

    import time
    s = time.time()
    runModel(0)
