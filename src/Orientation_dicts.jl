orientation_enum_to_string = Dict(
# ORIENTATION_RIP=>"RIP",
# ORIENTATION_LIP=>"LIP",
# ORIENTATION_RSP=>"RSP",
# ORIENTATION_LSP=>"LSP",
# ORIENTATION_RIA=>"RIA",
# ORIENTATION_LIA=>"LIA",
# ORIENTATION_RSA=>"RSA",
# ORIENTATION_LSA=>"LSA",
# ORIENTATION_IRP=>"IRP",
# ORIENTATION_ILP=>"ILP",
# ORIENTATION_SRP=>"SRP",
# ORIENTATION_SLP=>"SLP",
# ORIENTATION_IRA=>"IRA",
# ORIENTATION_ILA=>"ILA",
# ORIENTATION_SRA=>"SRA",
# ORIENTATION_SLA=>"SLA",
ORIENTATION_RPI=>"RPI",
ORIENTATION_LPI=>"LPI",
ORIENTATION_RAI=>"RAI",
ORIENTATION_LAI=>"LAI",
ORIENTATION_RPS=>"RPS",
ORIENTATION_LPS=>"LPS",
ORIENTATION_RAS=>"RAS",
ORIENTATION_LAS=>"LAS",
# ORIENTATION_PRI=>"PRI",
# ORIENTATION_PLI=>"PLI",
# ORIENTATION_ARI=>"ARI",
# ORIENTATION_ALI=>"ALI",
# ORIENTATION_PRS=>"PRS",
# ORIENTATION_PLS=>"PLS",
# ORIENTATION_ARS=>"ARS",
# ORIENTATION_ALS=>"ALS",
# ORIENTATION_IPR=>"IPR",
# ORIENTATION_SPR=>"SPR",
# ORIENTATION_IAR=>"IAR",
# ORIENTATION_SAR=>"SAR",
# ORIENTATION_IPL=>"IPL",
# ORIENTATION_SPL=>"SPL",
# ORIENTATION_IAL=>"IAL",
# ORIENTATION_SAL=>"SAL",
# ORIENTATION_PIR=>"PIR",
# ORIENTATION_PSR=>"PSR",
# ORIENTATION_AIR=>"AIR",
# ORIENTATION_ASR=>"ASR",
# ORIENTATION_PIL=>"PIL",
# ORIENTATION_PSL=>"PSL",
# ORIENTATION_AIL=>"AIL",
# ORIENTATION_ASL=>"ASL"
)

string_to_orientation_enum = Dict(value => key for (key, value) in orientation_enum_to_string)

orientation_dict_enum_to_number=Dict(
    # ORIENTATION_ASR => (0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    # ,ORIENTATION_AIL => (0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    # ,ORIENTATION_PSR => (0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    # ,ORIENTATION_ASL => (0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    ORIENTATION_RAS => (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    # ,ORIENTATION_AIR => (0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    # ,ORIENTATION_LIP => (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)
    ,ORIENTATION_LAS => (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    # ,ORIENTATION_SPL => (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    # ,ORIENTATION_PLS => (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    # ,ORIENTATION_ARI => (0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    ,ORIENTATION_LPI => (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    ,ORIENTATION_RAI => (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
    # ,ORIENTATION_IRP => (0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0)
    # ,ORIENTATION_SRP => (0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    # ,ORIENTATION_ALI => (0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    # ,ORIENTATION_PRS => (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    # ,ORIENTATION_RSP => (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    # ,ORIENTATION_PRI => (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    # ,ORIENTATION_IRA => (0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0)
    # ,ORIENTATION_SLA => (0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0)
    # ,ORIENTATION_IAR => (0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0)
    # ,ORIENTATION_PIR => (0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    # ,ORIENTATION_ILP => (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0)
    # ,ORIENTATION_ALS => (0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    # ,ORIENTATION_PIL => (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    # ,ORIENTATION_PSL => (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    # ,ORIENTATION_RSA => (-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
    # ,ORIENTATION_IPR => (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
    # ,ORIENTATION_RIP => (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)
    ,ORIENTATION_LAI => (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
    # ,ORIENTATION_PLI => (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    # ,ORIENTATION_LSA => (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
    # ,ORIENTATION_SPR => (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    # ,ORIENTATION_IAL => (0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0)
    # ,ORIENTATION_SAL => (0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0)
    # ,ORIENTATION_ARS => (0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ,ORIENTATION_LPS => (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # ,ORIENTATION_LSP => (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    # ,ORIENTATION_RIA => (-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0)
    # ,ORIENTATION_ILA => (0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0)
    # ,ORIENTATION_SRA => (0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0)
    ,ORIENTATION_RPS => (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # ,ORIENTATION_SAR => (0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0)
    ,ORIENTATION_RPI => (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    # ,ORIENTATION_LIA => (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0)
    # ,ORIENTATION_IPL => (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
    # ,ORIENTATION_SLP => (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
)
number_to_enum_orientation_dict = Dict(value => key for (key, value) in orientation_dict_enum_to_number)


orientation_pair_to_operation_dict =Dict((ORIENTATION_RAI, ORIENTATION_LPS) => (Int64[], [1, 2, 3], [[1, 1, 1, -1], [2, 2, 2, -1], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RPI, ORIENTATION_RPS) => (Int64[], [3], [[1, 3, 1, 0], [1, 3, 2, 0], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_LPI, ORIENTATION_LAS) => (Int64[], [2, 3], [[1, 3, 1, 0], [2, 2, 2, 1], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_LAS, ORIENTATION_RPI) => (Int64[], [1, 2, 3], [[1, 1, 1, 1], [2, 2, 2, -1], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_LAS, ORIENTATION_RPS) => (Int64[], [1, 2], [[1, 1, 1, 1], [2, 2, 2, -1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAI, ORIENTATION_RAI) => (Int64[], [1], [[1, 1, 1, 1], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAI, ORIENTATION_LPI) => (Int64[], [2], [[1, 3, 1, 0], [2, 2, 2, -1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LPI, ORIENTATION_RAS) => (Int64[], [1, 2, 3], [[1, 1, 1, 1], [2, 2, 2, 1], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RPS, ORIENTATION_RAI) => (Int64[], [2, 3], [[1, 3, 1, 0], [2, 2, 2, 1], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_LPS, ORIENTATION_LAI) => (Int64[], [2, 3], [[1, 3, 1, 0], [2, 2, 2, 1], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_RAI, ORIENTATION_LAS) => (Int64[], [1, 3], [[1, 1, 1, -1], [1, 3, 2, 0], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RPS, ORIENTATION_LPI) => (Int64[], [1, 3], [[1, 1, 1, -1], [1, 3, 2, 0], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_LAI, ORIENTATION_RPI) => (Int64[], [1, 2], [[1, 1, 1, 1], [2, 2, 2, -1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RPS, ORIENTATION_RPI) => (Int64[], [3], [[1, 3, 1, 0], [1, 3, 2, 0], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_RAI, ORIENTATION_RAS) => (Int64[], [3], [[1, 3, 1, 0], [1, 3, 2, 0], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_LAI, ORIENTATION_RPS) => (Int64[], [1, 2, 3], [[1, 1, 1, 1], [2, 2, 2, -1], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RPS, ORIENTATION_RPS) => (Int64[], Int64[], [[1, 3, 1, 0], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LPS, ORIENTATION_RAI) => (Int64[], [1, 2, 3], [[1, 1, 1, 1], [2, 2, 2, 1], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_LPS, ORIENTATION_LPI) => (Int64[], [3], [[1, 3, 1, 0], [1, 3, 2, 0], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_LPI, ORIENTATION_LAI) => (Int64[], [2], [[1, 3, 1, 0], [2, 2, 2, 1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RAS, ORIENTATION_LPS) => (Int64[], [1, 2], [[1, 1, 1, -1], [2, 2, 2, -1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RPI, ORIENTATION_LPS) => (Int64[], [1, 3], [[1, 1, 1, -1], [1, 3, 2, 0], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_LPS, ORIENTATION_RPI) => (Int64[], [1, 3], [[1, 1, 1, 1], [1, 3, 2, 0], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_RAI, ORIENTATION_LAI) => (Int64[], [1], [[1, 1, 1, -1], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LPS, ORIENTATION_RPS) => (Int64[], [1], [[1, 1, 1, 1], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAS, ORIENTATION_LPS) => (Int64[], [2], [[1, 3, 1, 0], [2, 2, 2, -1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RAS, ORIENTATION_LAS) => (Int64[], [1], [[1, 1, 1, -1], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LPI, ORIENTATION_RAI) => (Int64[], [1, 2], [[1, 1, 1, 1], [2, 2, 2, 1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LPI, ORIENTATION_LPI) => (Int64[], Int64[], [[1, 3, 1, 0], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RPI, ORIENTATION_LAS) => (Int64[], [1, 2, 3], [[1, 1, 1, -1], [2, 2, 2, 1], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RAS, ORIENTATION_RAS) => (Int64[], Int64[], [[1, 3, 1, 0], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LPI, ORIENTATION_RPI) => (Int64[], [1], [[1, 1, 1, 1], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAS, ORIENTATION_LAS) => (Int64[], Int64[], [[1, 3, 1, 0], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RAI, ORIENTATION_RAI) => (Int64[], Int64[], [[1, 3, 1, 0], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAI, ORIENTATION_LPS) => (Int64[], [2, 3], [[1, 3, 1, 0], [2, 2, 2, -1], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RPI, ORIENTATION_RAS) => (Int64[], [2, 3], [[1, 3, 1, 0], [2, 2, 2, 1], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_LPI, ORIENTATION_RPS) => (Int64[], [1, 3], [[1, 1, 1, 1], [1, 3, 2, 0], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RAI, ORIENTATION_LPI) => (Int64[], [1, 2], [[1, 1, 1, -1], [2, 2, 2, -1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RPS, ORIENTATION_LPS) => (Int64[], [1], [[1, 1, 1, -1], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAS, ORIENTATION_RAS) => (Int64[], [1], [[1, 1, 1, 1], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RAI, ORIENTATION_RPI) => (Int64[], [2], [[1, 3, 1, 0], [2, 2, 2, -1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAI, ORIENTATION_LAS) => (Int64[], [3], [[1, 3, 1, 0], [1, 3, 2, 0], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RAI, ORIENTATION_RPS) => (Int64[], [2, 3], [[1, 3, 1, 0], [2, 2, 2, -1], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RAS, ORIENTATION_LAI) => (Int64[], [1, 3], [[1, 1, 1, -1], [1, 3, 2, 0], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_RPS, ORIENTATION_LAS) => (Int64[], [1, 2], [[1, 1, 1, -1], [2, 2, 2, 1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LPS, ORIENTATION_LPS) => (Int64[], Int64[], [[1, 3, 1, 0], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RPI, ORIENTATION_LAI) => (Int64[], [1, 2], [[1, 1, 1, -1], [2, 2, 2, 1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAI, ORIENTATION_RAS) => (Int64[], [1, 3], [[1, 1, 1, 1], [1, 3, 2, 0], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_RPS, ORIENTATION_RAS) => (Int64[], [2], [[1, 3, 1, 0], [2, 2, 2, 1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAS, ORIENTATION_LAI) => (Int64[], [3], [[1, 3, 1, 0], [1, 3, 2, 0], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_RAS, ORIENTATION_RAI) => (Int64[], [3], [[1, 3, 1, 0], [1, 3, 2, 0], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_LPS, ORIENTATION_LAS) => (Int64[], [2], [[1, 3, 1, 0], [2, 2, 2, 1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RAS, ORIENTATION_LPI) => (Int64[], [1, 2, 3], [[1, 1, 1, -1], [2, 2, 2, -1], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_RPI, ORIENTATION_RAI) => (Int64[], [2], [[1, 3, 1, 0], [2, 2, 2, 1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RPI, ORIENTATION_LPI) => (Int64[], [1], [[1, 1, 1, -1], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LPI, ORIENTATION_LPS) => (Int64[], [3], [[1, 3, 1, 0], [1, 3, 2, 0], [3, 3, 3, -1]], [1, 2, 3])
,(ORIENTATION_LAI, ORIENTATION_LAI) => (Int64[], Int64[], [[1, 3, 1, 0], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LPS, ORIENTATION_RAS) => (Int64[], [1, 2], [[1, 1, 1, 1], [2, 2, 2, 1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RAS, ORIENTATION_RPI) => (Int64[], [2, 3], [[1, 3, 1, 0], [2, 2, 2, -1], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_LAS, ORIENTATION_LPI) => (Int64[], [2, 3], [[1, 3, 1, 0], [2, 2, 2, -1], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_RPS, ORIENTATION_LAI) => (Int64[], [1, 2, 3], [[1, 1, 1, -1], [2, 2, 2, 1], [3, 3, 3, 1]], [1, 2, 3])
,(ORIENTATION_RAS, ORIENTATION_RPS) => (Int64[], [2], [[1, 3, 1, 0], [2, 2, 2, -1], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_RPI, ORIENTATION_RPI) => (Int64[], Int64[], [[1, 3, 1, 0], [1, 3, 2, 0], [1, 3, 3, 0]], [1, 2, 3])
,(ORIENTATION_LAS, ORIENTATION_RAI) => (Int64[], [1, 3], [[1, 1, 1, 1], [1, 3, 2, 0], [3, 3, 3, 1]], [1, 2, 3]))


















