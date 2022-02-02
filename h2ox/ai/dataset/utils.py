import gcsfs


def gcsfs_mapper():
    fs = gcsfs.GCSFileSystem(access="read_only")
    return fs.get_mapper


def null_mapper():
    def return_null(x):
        return x

    return return_null
