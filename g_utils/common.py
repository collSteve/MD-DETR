def stardardize_object_class_name(raw_class_name: str) -> str:
    return f"cls_{raw_class_name}"

def unstandardize_object_class_name(standardized_class_name: str) -> str:
    if standardized_class_name.startswith("cls_"):
        return standardized_class_name[4:]
    return standardized_class_name