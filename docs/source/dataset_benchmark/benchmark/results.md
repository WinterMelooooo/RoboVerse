# Benchmark Results

## In-distribution Evaluation

<table>
  <thead>
    <tr>
      <th rowspan="2">Task name</th>
      <th>RGB</th>
      <th>RGBD</th>
      <th>PointCloud</th>
    </tr>
    <tr>
      <th>resnet18</th>
      <th>resnet18</th>
      <th>pointnet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CloseBoxL0</td>
      <td>0.81</td>
      <td>0.91</td>
      <td>0.82</td>
    </tr>
    <tr>
      <td>CloseBoxL1</td>
      <td>0.4</td>
      <td>0.58</td>
      <td>0.73</td>
    </tr>
    <tr>
      <td>CloseBoxL2</td>
      <td>0.42</td>
      <td>0.30</td>
      <td>0.34</td>
    </tr>
    <tr>
      <td>StackCubeL0</td>
      <td>0.91</td>
      <td>0.87</td>
      <td>0.73</td>
    </tr>
  </tbody>
</table>

## Out-of-distribution Evaluation(Zero-shot)
