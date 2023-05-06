from ECMIFrameWork import ECMIFrameWork #need to be modified according to your path
from util.config import ModelConf #need to be modified according to your path
config = ModelConf("/home/xxx/algor_name.conf")
rec = ECMIFrameWork(config)
rec.execute()
