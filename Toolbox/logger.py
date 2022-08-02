
import datetime
import logging
# 输出到console
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
# 输出到file
fh = logging.FileHandler("log_%s.txt"%(datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
fh.setLevel(logging.DEBUG)
        
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.addHandler(fh)
 
 
for i in range(10):
    logger.info(i)
