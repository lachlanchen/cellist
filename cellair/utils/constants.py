import os


curdir = os.getcwd()
print("curdir: ", curdir)


# dataroot = "/home/lachlan/Projects/www_hhp/data/"
dataroot = os.path.join(curdir, "data")
# dataroot_remote = os.path.join(curdir, "data-remote")

mysqlconfig = {
    "host": "localhost",
    "user": "root",
    "password": "lazeal0626"

}
mysqlurl = f'mysql+pymysql://{mysqlconfig["user"]}:{mysqlconfig["password"]}@{mysqlconfig["host"]}/cellair'
