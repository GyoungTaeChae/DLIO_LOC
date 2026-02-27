This is forked version of direct_lidar_inertial_odometry

### Localization
localization against a pre-built PCD map. First, save a map using the odometry mode.(refer to original repo) then run:

```sh
roslaunch direct_lidar_inertial_odometry dlio_loc.launch
```

Configure the global map path and initial pose in `cfg/loc_params.yaml`:

```yaml
dlio:
  loc:
    globalmap:
      pcd_path: "/path/to/map.pcd"
      leaf_size: 0.5        # map downsample resolution (0 = no downsample)

    initial_pose:
      use_rviz: false       # true: set initial pose via RViz "2D Pose Estimate"
      use_yaml: true        # true: use yaml values below
      x: 0.0
      y: 0.0
      z: 0.0
      roll: 0.0             # radians
      pitch: 0.0            # radians
      yaw: 0.0              # radians (e.g. 3.14159 for 180 degrees)
```
