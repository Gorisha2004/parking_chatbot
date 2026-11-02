from parking_chatbot import config


def test_static_data_dir_is_inside_the_project_root():
    assert config.STATIC_DATA_DIR == config.PROJECT_ROOT / "data" / "static"


def test_chunk_overlap_is_smaller_than_chunk_size():
    assert 0 < config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_max_advance_days_is_a_positive_integer():
    assert isinstance(config.MAX_ADVANCE_DAYS, int)
    assert config.MAX_ADVANCE_DAYS > 0


def test_top_k_is_a_positive_integer():
    assert isinstance(config.TOP_K, int)
    assert config.TOP_K > 0
