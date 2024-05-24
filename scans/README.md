# Phantom Scans

This folder contains all image data used in the paper. The [240304 dataset](240304) was acquired in one 3-hour scan session on a 3.0 T GE Premier MRI Scanner. The following sections provide practical guidance for reproducing the data or using the same scanning paradigm to conduct new experiments.

## Phantom Configuration Order

During the scan session the phantom was re-configured in situ (on the scan table) five times through the following sequence of configurations:

1. Structured signal, plastic implant replica
2. Structured signal, metal implant
3. Uniform signal, no implant
4. Uniform signal, plastic implant replica
5. Uniform signal, metal implant
6. Structured signal, plastic implant replica

A few notes about the order:
- Grouping configurations by structure helps to streamline the re-configuration stages
- Bookending the scan session with the Structured-Plastic configuration is helpful for checking net displacement of phantom components over the course of the scan session.
- It was found empirically that starting the scan session with a metal configuration produced additional artifacts in all subsequent scans with any configuration. It is hypothesized that a one-time pre-scan process was getting confounded by the metal configuration of this phantom. In any case, the issue is easily avoided by beginning the scan session with a plastic configuration. Your mileage may vary on a different scanner.

## Phantom & Coil Arrangement

Phantom and receive coils should be arranged on the patient table with three goals in mind:
1. Emulate the in vivo context.
2. Facilitate in situ re-configuration of the phantom.
3. Maximize positional consistency between phantom configurations.

A few tips for achieving these goals:
1. Rest the phantom on a hard surface (i.e. no table padding). This will ensure the phantom does not shift under the different/uneven weighting of metal and plastic configurations.
2. If a table coil will be used, consider elevating the phantom for a more realistic implant placement (and more uniform coil sensitivity).
3. If an external coil will be used, avoid placing the coil in direct contact with the phantom. In the paper this was done by resting a flexible abdominal coil on a plastic "rib-cage" mount so that it could be slid on/off without moving the phantom. This setup made it easy to reconfigure the phantom in situ. The landmarking light can be used to replace the coil with reasonable repeatability.
4. Leave the lid of the phantom container unlatched for ease of removal/replacement before/after reconfiguring the phantom.

## Pulse Sequence Protocols

Refer to section 2.2 (Data Acquisition) of the TONIQ paper for a description of the pulse sequence protocols used to acquire this dataset.
