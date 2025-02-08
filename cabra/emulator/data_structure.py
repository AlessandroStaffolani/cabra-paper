from cabra.common.enum_utils import ExtendedEnum


class ModelTypes(str, ExtendedEnum):
    SyntheticDataGenerator = 'synthetic-data'
    RealData = 'real-data'
    CDRCData = 'cdrc-data'
    TestData = 'test-data'
