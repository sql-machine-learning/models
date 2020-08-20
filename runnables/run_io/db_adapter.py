import re

def parseMySQLDSN(dsn):
    # [username[:password]@][protocol[(address)]]/dbname[?param1=value1&...&paramN=valueN]
    pattern = "^(\w*):(\w*)@tcp\(([.a-zA-Z0-9\-]*):([0-9]*)\)/(\w*)(\?.*)?$"  # noqa: W605, E501
    found_result = re.findall(pattern, dsn)
    user, passwd, host, port, database, config_str = found_result[0]
    config = {}
    if len(config_str) > 1:
        for c in config_str[1:].split("&"):
            k, v = c.split("=")
            config[k] = v
    return user, passwd, host, port, database, config

# TODO(brightcoder01): Should we put this kind of common method
# in sqlflow runtime? While writing the runnable code, users can
# import the runtime library.
def convertDSNToRfc1738(driver_dsn, defaultDbName):
    driver, dsn = driver_dsn.split("://")
    user, passwd, host, port, database, config = parseMySQLDSN(dsn)

    if not database:
        database = defaultDbName

    # mysql://root:root@127.0.0.1:3306/dbname
    return "{}://{}:{}@{}:{}/{}".format(driver, user, passwd, host, port, database)
