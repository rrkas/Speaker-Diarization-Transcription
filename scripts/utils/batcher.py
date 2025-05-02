def batchify(_lst: list, batch_size: int):
    batch_size = int(batch_size)

    for i in range(0, len(_lst), batch_size):
        yield _lst[i : i + batch_size]
