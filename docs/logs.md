2026-01-03T15:11:01.1279106Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1280212Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1281006Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1281447Z 
2026-01-03T15:11:01.1281538Z Stacktrace:
2026-01-03T15:11:01.1281793Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1282570Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1283141Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1283643Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1284143Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1284751Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1286474Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1287837Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1288503Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(101, 534, 532), MedImages=(95, 534, 538)
2026-01-03T15:11:01.1289232Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1289789Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1290037Z    Evaluated: (101, 534, 532) == (95, 534, 538)
2026-01-03T15:11:01.1290202Z 
2026-01-03T15:11:01.1290281Z Stacktrace:
2026-01-03T15:11:01.1290483Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1291076Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1291599Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1292071Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1292496Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1293076Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1294671Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1295978Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1296731Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1297369Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1297927Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1298245Z 
2026-01-03T15:11:01.1298641Z Stacktrace:
2026-01-03T15:11:01.1298935Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1299576Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1300115Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1300580Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1301006Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1301575Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1303130Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1304445Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1305109Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(107, 540, 537), MedImages=(100, 540, 544)
2026-01-03T15:11:01.1305819Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1306294Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1306537Z    Evaluated: (107, 540, 537) == (100, 540, 544)
2026-01-03T15:11:01.1306700Z 
2026-01-03T15:11:01.1306771Z Stacktrace:
2026-01-03T15:11:01.1306974Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1307548Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1308086Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1308542Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1308973Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1309736Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1311318Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1312625Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1313398Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1314031Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1314582Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1314893Z 
2026-01-03T15:11:01.1314974Z Stacktrace:
2026-01-03T15:11:01.1315177Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1315764Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1316280Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1316848Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1317342Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1317933Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1319690Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1321016Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1321675Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(101, 534, 532), MedImages=(95, 534, 538)
2026-01-03T15:11:01.1322401Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1322873Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1323109Z    Evaluated: (101, 534, 532) == (95, 534, 538)
2026-01-03T15:11:01.1323269Z 
2026-01-03T15:11:01.1323334Z Stacktrace:
2026-01-03T15:11:01.1323537Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1324110Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1324622Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1325079Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1325497Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1326064Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1327626Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1328922Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1329864Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1330508Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1331057Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1331376Z 
2026-01-03T15:11:01.1331439Z Stacktrace:
2026-01-03T15:11:01.1331643Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1332205Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1332732Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1333184Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1333610Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1334169Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1335826Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1337181Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1337840Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(107, 540, 537), MedImages=(100, 540, 544)
2026-01-03T15:11:01.1338544Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1339024Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1339261Z    Evaluated: (107, 540, 537) == (100, 540, 544)
2026-01-03T15:11:01.1339490Z 
2026-01-03T15:11:01.1339556Z Stacktrace:
2026-01-03T15:11:01.1339760Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1340349Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1340881Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1341338Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1341766Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1342333Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1343900Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1345196Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1345963Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1346584Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1347132Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1347438Z 
2026-01-03T15:11:01.1347502Z Stacktrace:
2026-01-03T15:11:01.1347705Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1348277Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1348793Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1349256Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1349826Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1350401Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1351964Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1353266Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1353998Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(101, 534, 532), MedImages=(95, 534, 538)
2026-01-03T15:11:01.1354764Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1355236Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1355471Z    Evaluated: (101, 534, 532) == (95, 534, 538)
2026-01-03T15:11:01.1355629Z 
2026-01-03T15:11:01.1355702Z Stacktrace:
2026-01-03T15:11:01.1355897Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1356476Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1356988Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1357460Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1357883Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1358455Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1360082Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1361388Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1362139Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1362775Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1363315Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1363635Z 
2026-01-03T15:11:01.1363701Z Stacktrace:
2026-01-03T15:11:01.1363907Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1364468Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1364994Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1365455Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1365877Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1366434Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1368000Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1369301Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1370018Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(107, 540, 537), MedImages=(100, 540, 544)
2026-01-03T15:11:01.1370723Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1371197Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1371427Z    Evaluated: (107, 540, 537) == (100, 540, 544)
2026-01-03T15:11:01.1371598Z 
2026-01-03T15:11:01.1371664Z Stacktrace:
2026-01-03T15:11:01.1371864Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1372505Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1373085Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1373539Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1373976Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1374544Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1376116Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1377441Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1378200Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1378824Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1379426Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1379734Z 
2026-01-03T15:11:01.1379819Z Stacktrace:
2026-01-03T15:11:01.1380012Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1380588Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1381103Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1381567Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1381992Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1382571Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1384131Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1385423Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1386073Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(101, 534, 532), MedImages=(95, 534, 538)
2026-01-03T15:11:01.1386785Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1387250Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1387489Z    Evaluated: (101, 534, 532) == (95, 534, 538)
2026-01-03T15:11:01.1387650Z 
2026-01-03T15:11:01.1387718Z Stacktrace:
2026-01-03T15:11:01.1387906Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1388475Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1388993Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1389578Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1390096Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1390742Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1392362Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1393663Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1394420Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1395052Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1395593Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1395908Z 
2026-01-03T15:11:01.1395972Z Stacktrace:
2026-01-03T15:11:01.1396170Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1396733Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1397250Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1397700Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1398123Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1398691Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1400295Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1401600Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1402259Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(107, 540, 537), MedImages=(100, 540, 544)
2026-01-03T15:11:01.1402963Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1403443Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1403683Z    Evaluated: (107, 540, 537) == (100, 540, 544)
2026-01-03T15:11:01.1403844Z 
2026-01-03T15:11:01.1403914Z Stacktrace:
2026-01-03T15:11:01.1404122Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1404694Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1405213Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1405675Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1406113Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1406682Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1408308Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1409729Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1410484Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1411104Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1411650Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1411954Z 
2026-01-03T15:11:01.1412027Z Stacktrace:
2026-01-03T15:11:01.1412219Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1412786Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1413305Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1413773Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1414202Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1414763Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1416312Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1417611Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1418258Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(101, 534, 532), MedImages=(95, 534, 538)
2026-01-03T15:11:01.1418958Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1419482Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1419718Z    Evaluated: (101, 534, 532) == (95, 534, 538)
2026-01-03T15:11:01.1419877Z 
2026-01-03T15:11:01.1419951Z Stacktrace:
2026-01-03T15:11:01.1420141Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1420712Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1421227Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1421688Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1422117Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1422683Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1424248Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1425541Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1426296Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1426985Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1427808Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1428267Z 
2026-01-03T15:11:01.1428333Z Stacktrace:
2026-01-03T15:11:01.1428547Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1429120Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1429779Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1430232Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1430660Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1431232Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1432790Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1434089Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1434747Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(107, 540, 537), MedImages=(100, 540, 544)
2026-01-03T15:11:01.1435453Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1435930Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1436168Z    Evaluated: (107, 540, 537) == (100, 540, 544)
2026-01-03T15:11:01.1436333Z 
2026-01-03T15:11:01.1436401Z Stacktrace:
2026-01-03T15:11:01.1436604Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1437166Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1437688Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1438137Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1438568Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1439138Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1440779Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1442081Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1442838Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1443460Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1444011Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1444317Z 
2026-01-03T15:11:01.1444394Z Stacktrace:
2026-01-03T15:11:01.1444593Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1445253Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1445870Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1446338Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1446758Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1447323Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1448872Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1450242Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1450896Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(101, 534, 532), MedImages=(95, 534, 538)
2026-01-03T15:11:01.1451608Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1452084Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1452312Z    Evaluated: (101, 534, 532) == (95, 534, 538)
2026-01-03T15:11:01.1452470Z 
2026-01-03T15:11:01.1452545Z Stacktrace:
2026-01-03T15:11:01.1452739Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1453311Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1453824Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1454286Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1454709Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1455267Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1456820Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1458116Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1458860Z Metadata Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:90[22m
2026-01-03T15:11:01.1459549Z   Expression: isapprox(collect(sitk_image.GetOrigin()), collect(med_im.origin); atol = origin_atol)
2026-01-03T15:11:01.1460099Z    Evaluated: isapprox([-179.93124389648438, 187.03125, -433.0], [-179.93124389648438, 171.5625, -433.0]; atol = 0.1)
2026-01-03T15:11:01.1460402Z 
2026-01-03T15:11:01.1460466Z Stacktrace:
2026-01-03T15:11:01.1460675Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1461239Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1461761Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1462222Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:90[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1462642Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1463307Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1464904Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1466236Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:88[24m[39m
2026-01-03T15:11:01.1466900Z [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mDimension mismatch: SimpleITK=(107, 540, 537), MedImages=(100, 540, 544)
2026-01-03T15:11:01.1467608Z Voxel Data Comparison: [91m[1mTest Failed[22m[39m at [39m[1m/home/runner/work/MedImages.jl/MedImages.jl/test/test_helpers.jl:117[22m
2026-01-03T15:11:01.1468092Z   Expression: sitk_dims == med_dims
2026-01-03T15:11:01.1468341Z    Evaluated: (107, 540, 537) == (100, 540, 544)
2026-01-03T15:11:01.1468510Z 
2026-01-03T15:11:01.1468577Z Stacktrace:
2026-01-03T15:11:01.1468775Z  [1] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1469338Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:672[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1470043Z  [2] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1470510Z [90m   @[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:117[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1470936Z  [3] [0m[1mmacro expansion[22m
2026-01-03T15:11:01.1471511Z [90m   @[39m [90m/opt/hostedtoolcache/julia/1.10.10/x64/share/julia/stdlib/v1.10/Test/src/[39m[90m[4mTest.jl:1577[24m[39m[90m [inlined][39m
2026-01-03T15:11:01.1473091Z  [4] [0m[1mtest_object_equality[22m[0m[1m([22m[90mmed_im[39m::[0mMedImage, [90msitk_image[39m::[0mPyObject; [90mspacing_atol[39m::[0mFloat64, [90mdirection_atol[39m::[0mFloat64, [90morigin_atol[39m::[0mFloat64, [90mvoxel_rtol[39m::[0mFloat64, [90mvoxel_atol[39m::[0mFloat64, [90mallow_dimension_mismatch[39m::[0mBool, [90mskip_voxel_comparison[39m::[0mBool[0m[1m)[22m
2026-01-03T15:11:01.1474397Z [90m   @[39m [35mMain.TestHelpers[39m [90m~/work/MedImages.jl/MedImages.jl/test/[39m[90m[4mtest_helpers.jl:94[24m[39m
2026-01-03T15:11:01.1475533Z WARNING: Method definition get_sitk_interpolator(Any) in module Main at /home/runner/work/MedImages.jl/MedImages.jl/test/basic_transformations_tests/test_rotate_mi.jl:70 overwritten at /home/runner/work/MedImages.jl/MedImages.jl/test/basic_transformations_tests/test_scale_mi.jl:25.
2026-01-03T15:11:01.1477180Z WARNING: Method definition get_sitk_interpolator(Any) in module Main at /home/runner/work/MedImages.jl/MedImages.jl/test/basic_transformations_tests/test_scale_mi.jl:25 overwritten at /home/runner/work/MedImages.jl/MedImages.jl/test/spatial_metadata_change_tests/test_resample_to_spacing.jl:18.
2026-01-03T15:14:10.7981122Z 
2026-01-03T15:14:10.7991665Z [2621] signal (15): Terminated
2026-01-03T15:14:10.8000178Z in expression starting at /home/runner/work/_actions/julia-actions/julia-runtest/v1/test_harness.jl:62
2026-01-03T15:14:10.8219709Z ##[error]The runner has received a shutdown signal. This can happen when the runner service is stopped, or a manually started runner is canceled.
2026-01-03T15:14:10.8315914Z epoll_wait at /lib/x86_64-linux-gnu/libc.so.6 (unknown line)
2026-01-03T15:14:10.9000480Z uv__io_poll at /workspace/srcdir/libuv/src/unix/epoll.c:236
2026-01-03T15:14:10.9206096Z uv_run at /workspace/srcdir/libuv/src/unix/core.c:400
2026-01-03T15:14:10.9543848Z ijl_task_get_next at /cache/build/builder-amdci5-7/julialang/julia-release-1-dot-10/src/partr.c:478
2026-01-03T15:14:12.6942789Z ##[error]The operation was canceled.
2026-01-03T15:14:12.7552413Z Cleaning up orphan processes