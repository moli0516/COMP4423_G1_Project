SPLIT_NAMES = ("train", "validation", "test")
CLASS_NAMES = {0: "outside", 1: "inside"}
REQUIRED_IMAGE_COLUMNS = {"image_id", "relative_path", "label", "group_id", "split"}
REQUIRED_LOCATION_COLUMNS = {"group_id", "class_name", "boundary_verified"}
