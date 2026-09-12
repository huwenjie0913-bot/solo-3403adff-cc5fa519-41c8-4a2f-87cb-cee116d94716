import os
import tempfile

# 必须在导入 app 之前设置，保证测试使用独立数据库
os.environ["ANNEALING_DB"] = os.path.join(tempfile.mkdtemp(prefix="annealing_test_"), "test.db")
