# R9 Full failure diagnostic

The archived exact-one gate fails on four union sample IDs. This diagnostic reproduces the same detector-count pattern twice per source/native/candidate image with the locked buffalo_l, 224x224 CUDA analyzer. It does not change or replace the formal evaluator.

- `val:Manually_Annotated_Images/680/a307c74778e9d37808732047d80fa7b3f2db2b17f289e42002d7cbc6.jpg`: archived counts source/native/candidate = 1/2/1; diagnostic repeat counts match.
- `val:Manually_Annotated_Images/365/321800005c95997baeb50709fcb9450e44196a76950674c10b57d16b.jpg`: archived counts source/native/candidate = 1/1/2; diagnostic repeat counts match.
- `val:Manually_Annotated_Images/804/9c405664818542b6d5340e71b5c5d84565d6ca1665f643a89fe30cc1.JPG`: archived counts source/native/candidate = 1/2/1; diagnostic repeat counts match.
- `val:Manually_Annotated_Images/627/a4c5c6830c879c675a70f3129ba2be3232df5f574aed73a2ef9689a4.jpg`: archived counts source/native/candidate = 1/2/1; diagnostic repeat counts match.

The existing 2048 paired sharpness rows were analyzed without rerunning FID or KID. The overall candidate-minus-native mean is -95.514784, and the candidate is sharper in 852 of 2048 pairs. See `sharpness_deciles.csv` for the native-sharpness-ranked deciles.
