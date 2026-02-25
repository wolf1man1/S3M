
# Band Selection
# Keep bands 0-109 (Indices 0-109)
SELECTED_BANDS = list(range(110))

# Wavelengths for the selected bands (Indices 0-109)
# Extracted from s185_band_lookup (1).csv
WAVELENGTHS = [
    450.0, 454.03, 458.06, 462.10, 466.13, 470.16, 474.19, 478.23, 482.26, 486.29,
    490.32, 494.35, 498.39, 502.42, 506.45, 510.48, 514.52, 518.55, 522.58, 526.61,
    530.65, 534.68, 538.71, 542.74, 546.77, 550.81, 554.84, 558.87, 562.90, 566.94,
    570.97, 575.00, 579.03, 583.06, 587.10, 591.13, 595.16, 599.19, 603.23, 607.26,
    611.29, 615.32, 619.35, 623.39, 627.42, 631.45, 635.48, 639.52, 643.55, 647.58,
    651.61, 655.65, 659.68, 663.71, 667.74, 671.77, 675.81, 679.84, 683.87, 687.90,
    691.94, 695.97, 700.00, 704.03, 708.06, 712.10, 716.13, 720.16, 724.19, 728.23,
    732.26, 736.29, 740.32, 744.35, 748.39, 752.42, 756.45, 760.48, 764.52, 768.55,
    772.58, 776.61, 780.65, 784.68, 788.71, 792.74, 796.77, 800.81, 804.84, 808.87,
    812.90, 816.94, 820.97, 825.00, 829.03, 833.06, 837.10, 841.13, 845.16, 849.19,
    853.23, 857.26, 861.29, 865.32, 869.35, 873.39, 877.42, 881.45, 885.48, 889.52
]

# Indices for VI Calculation (Approximate based on wavelengths)
# Blue: ~450nm (Index 0) - 490nm (Index 10) -> Let's pick 470nm (Index 5)
# Green: ~550nm (Index 25)
# Red: ~650nm (Index 50) - 670nm (Index 55) -> Let's pick 660nm (Index 52)
# RedEdge: ~700nm (Index 62) - 740nm (Index 72) -> Let's pick 720nm (Index 67)
# NIR: ~800nm (Index 87) - 850nm (Index 100) -> Let's pick 833nm (Index 95)

IDX_BLUE = 5      # 470nm
IDX_GREEN = 25    # 550nm
IDX_RED = 52      # 660nm
IDX_RED_EDGE = 67 # 720nm
IDX_NIR = 95      # 833nm

# Class Mapping
CLASS_MAP = {
    'Health': 0,
    'Rust': 1,
    'Other': 2
}

# Image Statistics (Approximate max value for normalization)
MAX_PIXEL_VAL = 10000.0

# Blacklist (All Black Images provided by user)
BLACKLIST_FILES = [
    "Health_hyper_167.tif", "Health_hyper_26.tif", "Other_hyper_22.tif", "Health_hyper_12.tif",
    "Health_hyper_23.tif", "Other_hyper_149.tif", "Health_hyper_76.tif", "Health_hyper_38.tif",
    "Other_hyper_174.tif", "Health_hyper_34.tif", "Other_hyper_122.tif", "Health_hyper_153.tif",
    "Other_hyper_113.tif", "Other_hyper_121.tif", "Other_hyper_64.tif", "Other_hyper_163.tif",
    "Health_hyper_67.tif", "Other_hyper_102.tif", "Other_hyper_31.tif", "Other_hyper_155.tif",
    "Other_hyper_50.tif", "Other_hyper_26.tif", "Other_hyper_160.tif"
]
