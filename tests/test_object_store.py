from bam_kv_cache.ssd.object_store import FileObjectStore, SsdLocation


def test_alloc_is_aligned(tmp_path):
    store = FileObjectStore(str(tmp_path), block_alignment=4096)
    a = store.allocate(100)
    b = store.allocate(100)
    assert a.offset == 0
    assert b.offset == 4096  # second alloc starts on next aligned block
    assert a.object_id != b.object_id


def test_write_read_roundtrip(tmp_path):
    store = FileObjectStore(str(tmp_path))
    loc = store.allocate(8)
    store.write(loc, b"12345678")
    assert store.read(loc) == b"12345678"


def test_distinct_extents_do_not_overlap(tmp_path):
    store = FileObjectStore(str(tmp_path))
    a = store.allocate(8)
    b = store.allocate(8)
    store.write(a, b"aaaaaaaa")
    store.write(b, b"bbbbbbbb")
    assert store.read(a) == b"aaaaaaaa"
    assert store.read(b) == b"bbbbbbbb"


def test_write_size_mismatch_raises(tmp_path):
    store = FileObjectStore(str(tmp_path))
    loc = store.allocate(8)
    try:
        store.write(loc, b"123")
    except ValueError:
        return
    raise AssertionError("expected ValueError on size mismatch")
