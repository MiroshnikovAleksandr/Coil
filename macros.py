coils = [[0.475, 0.45],
         [0.47, 0.455],
         [0.46499999999999997, 0.46],
         [0.48, 0.432258064516129, 0.0831425598335067],
         [0.485, 0.41337148803329865, 0.18099375650364202],
         [0.49, 0.39658168574401664, 0.24170655567117583],
         [0.495, 0.3783246618106139, 0.28762747138397504],
         [0.5, 0.35132674297606653, 0.32168574401664934]]


def create_macros(coils):
    rad = 'Array(' + ', '.join(f'Array({", ".join(str(r) for r in coil)})' for coil in coils) + ')'

    with open('macros.cst', 'w') as f:
        f.write(f"""Option Explicit

Sub main
Dim rad() As Variant, r As Variant, i As Integer, j As Integer
rad = {rad}
For i = 0 To UBound(rad())
     For j = 0 To UBound(rad(i))
          r = rad(i)(j)
          With Arc
              .Reset
              .Name "arc" + Str(i) + Str(j)
              .Curve "curve1"
              .Orientation "Clockwise"
              .XCenter "0"
              .YCenter "-0"
              .X1 r
              .Y1 "0.0"
              .X2 "0.0"
              .Y2 "0.0"
               .Angle "350"
              .UseAngle "True"
              .Segments "0"
              .Create
          End With
    Next j
Next i


For i = 1 To UBound(rad())
     For j = 0 To UBound(rad(i))
     With Transform
          .Reset
          .Name "curve1:arc" + Str(i) + Str(j)
          .Vector "0", "0", "0.003"
          .UsePickedPoints "False"
          .InvertPickedPoints "False"
          .MultipleObjects "False"
          .GroupObjects "False"
          .Repetitions i
          .MultipleSelection "False"
          .Transform "Curve", "Translate"
     End With
     Next j
Next i


For i = 0 To UBound(rad())
     For j = 0 To UBound(rad(i)) - 1
     With Polygon3D
        .Reset
        .Version 10
        .Name "3dpolygon" + Str(i) + Str(j)
        .Curve "curve1"
        .Point rad(i)(j), "0", i*0.003
        .Point rad(i)(j + 1)*Cos(pi/18), rad(i)(j + 1)*Sin(pi/18), i*0.003
        .Create
     End With
     Next j
Next i


	
End Sub""")


create_macros(coils)
