# Benchmark Results

## In-distribution Evaluation

<table>
  <thead>
    <tr>
      <th rowspan="2">Task name</th>
      <th>RGB</th>
      <th colspan="3"><center>RGBD</center></th>
      <th colspan="2">PointCloud</th>
    </tr>
    <tr>
      <th>resnet18</th>
      <th>resnet18</th>
      <th>ViT</th>
      <th>MultiViT</th>
      <th>pointnet</th>
      <th>spUnet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CloseBoxL0</td>
      <td>0.81</td>
      <td><b>0.91 </b></td>
      <td>0.89</td>
      <td>----</td>
      <td>0.82</td>
      <td>----</td>
    </tr>
    <tr>
      <td>CloseBoxL1</td>
      <td>0.4</td>
      <td>0.58</td>
      <td>0.48</td>
      <td>----</td>
      <td><b>0.73</b></td>
      <td>----</td>
    </tr>
    <tr>
      <td>CloseBoxL2</td>
      <td><b>0.42</b></td>
      <td>0.30</td>
      <td>----</td>
      <td>----</td>
      <td>0.34</td>
      <td>----</td>
    </tr>
    <tr>
      <td>StackCubeL0</td>
      <td><b>0.91</b></td>
      <td>0.87</td>
      <td>----</td>
      <td>----</td>
      <td>0.00</td>
      <td>0.06</td>
    </tr>
    <tr>
      <td>StackCubeL1</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>----</td>
      <td>----</td>
      <td>0.00</td>
      <td>----</td>
    </tr>
    <tr>
      <td>StackCubeL2</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>----</td>
      <td>----</td>
      <td>0.00</td>
      <td>----</td>
    </tr>
  </tbody>
</table>

## Out-of-distribution Evaluation(Zero-shot)
<table>
  <thead>
    <tr>
      <th rowspan="2">Task name</th>
      <th>RGB</th>
      <th colspan="3"><center>RGBD</center></th>
      <th colspan="2">PointCloud</th>
    </tr>
    <tr>
      <th>resnet18</th>
      <th>resnet18</th>
      <th>ViT</th>
      <th>MultiViT</th>
      <th>pointnet</th>
      <th>spUnet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CloseBoxL0</td>
      <td>0.52</td>
      <td><b>0.72</b></td>
      <td>----</td>
      <td>----</td>
      <td>0.60</td>
      <td>----</td>
    </tr>
    <tr>
      <td>CloseBoxL1</td>
      <td>0.20</td>
      <td>0.50</td>
      <td>----</td>
      <td>----</td>
      <td><b>0.77</b></td>
      <td>----</td>
    </tr>
    <tr>
      <td>CloseBoxL2</td>
      <td>0.32</td>
      <td><b>0.38</b></td>
      <td>----</td>
      <td>----</td>
      <td>0.38</td>
      <td>----</td>
    </tr>
    <tr>
      <td>StackCubeL0</td>
      <td><b>0.29</b></td>
      <td>0.19</td>
      <td>----</td>
      <td>----</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>

  </tbody>
</table>
