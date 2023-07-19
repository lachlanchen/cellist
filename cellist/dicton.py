class Dicton(object):
    instances = {}
    returned_exist = False
    def __new__(cls, uuid, *args, **kwargs):

        # print(uuid, args, kwargs)
        # print(model_id)
        uuid = f"{cls.__name__}-{uuid}"

        _instance = cls.instances.get(uuid, None)
        if _instance is None:
            _instance = super(Dicton, cls).__new__(cls)
            print(f"Create new instance with uuid={uuid}.")
            cls.instances[uuid] = _instance
            cls.returned_exist = False
        else:
            print(f"Return the instance with uuid={uuid}.")
            cls.returned_exist = True

        return _instance

class A(Dicton):
    pass



if __name__ == '__main__':
    s1 = A("aa")
    s2 = A("aa")
    if (id(s1) == id(s2)):
        print("Same")
    else:
        print("Different")

