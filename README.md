# Direct LiDAR-Inertial Odometry (Forked)

This is a forked version of [DLIO (Direct LiDAR-Inertial Odometry)](https://github.com/vectr-ucla/direct_lidar_inertial_odometry), with an added localization mode supporting GICP-based and NDT_OMP-based scan matching.

## Usage

### LIO (Odometry + Mapping)

```bash
roslaunch direct_lidar_inertial_odometry dlio.launch
```

### Localization

```bash
roslaunch direct_lidar_inertial_odometry dlio_loc.launch
```

The scan matching method can be selected in `cfg/loc_params.yaml`:

```yaml
dlio:
  loc:
    registration:
      method: "GICP"      # "GICP" or "NDT_OMP"
```
