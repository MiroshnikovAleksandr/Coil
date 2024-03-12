coils = [[0.013513513513513514, 0.09358108108108108, 0.09662162162162162, 0.09966216216216217, 0.10270270270270271,
         0.10675675675675676]]
coords_start = [[-0.5, -0.866025], [-1, 0], [-0.5, 0.866025], [0.5, 0.866025], [1, 0], [0.5, -0.866025]]
coords = [[[[-0.5, -0.866025], [-1.0, 0.0], [-0.5, 0.866025], [0.5, 0.866025], [1.0, 0.0], [0.5, -0.866025]]],
          [[[-0.375, -0.64951875], [-0.75, 0.0], [-0.375, 0.64951875], [0.375, 0.64951875], [0.75, 0.0],
            [0.375, -0.64951875]],
           [[-0.03333333333333333, -0.057735], [-0.06666666666666667, 0.0], [-0.03333333333333333, 0.057735],
            [0.03333333333333333, 0.057735], [0.06666666666666667, 0.0], [0.03333333333333333, -0.057735]],
           [[-0.016666666666666666, -0.0288675], [-0.03333333333333333, 0.0], [-0.016666666666666666, 0.0288675],
            [0.016666666666666666, 0.0288675], [0.03333333333333333, 0.0], [0.016666666666666666, -0.0288675]]]]


# s_1 = ''
# for seq in coords:
#     s_2 = ''
#     for turn in seq:
#         turn_str = 'Array(' + ', '.join(f'Array({", ".join(str(num) for num in pair)})' for pair in turn) + ')'
#         s_2 += turn_str + ', '
#     s_2 = s_2[:-2]
#     coord_a = 'Array(' + s_2 + ')'
#     s_1 += coord_a + ', '
# s_1 = s_1[:-2]
# array = 'Array(' + s_1 + ')'
# print(array)


def create_piecewise_macros(coords):
    s_1 = ''
    for seq in coords:
        s_2 = ''
        for turn in seq:
            turn_str = 'Array(' + ', '.join(f'Array({", ".join(str(num) for num in pair)})' for pair in turn) + ')'
            s_2 += turn_str + ', '
        s_2 = s_2[:-2]
        coord_a = 'Array(' + s_2 + ')'
        s_1 += coord_a + ', '
    s_1 = s_1[:-2]
    rad = 'Array(' + s_1 + ')'
    macros = f"""

    Sub main
    Dim rad() As Variant, r As Variant, i As Integer, j As Integer, k As Integer, gap As Variant
    rad = {rad}
    gap = 0.01
    For i = 0 To UBound(rad())
         For j = 0 To UBound(rad(i))
              r = rad(i)(j)
              With Polygon 
                   .Reset 
                   .Name "turn" + Str(i) + Str(j)
                   .Curve "curve1" 
                   .Point r(0)(0), r(0)(1)
                   For k = 1 To UBound(r)
                        .LineTo r(k)(0), r(k)(1)
                   Next k
                   .LineTo r(0)(0), r(0)(1) - gap
                   .Create 
              End With
        Next j
    Next i


    For i = 0 To UBound(rad())
         For j = 0 To UBound(rad(i)) - 1
         With Line
              .Reset
              .Name "connection" + Str(j) + "-" + Str(j + 1)
              .Curve "curve1" 
              .X1 rad(i)(j)(0)(0) 
              .Y1 rad(i)(j)(0)(1) - gap
              .X2 rad(i)(j+1)(0)(0) 
              .Y2 rad(i)(j+1)(0)(1) 
              .Create 
         End With
         Next j
    Next i

    
    For i = 0 To UBound(rad())
         For j = 0 To UBound(rad(i)) - 1
         With Polygon
            .Reset
            .Version 10
            .Name "3dpolygon" + Str(i) + Str(j)
            .Curve "curve1"
            .Point rad(i)(j), "0", "0"
            .Point rad(i)(j + 1)*Cos(pi/18), rad(i)(j + 1)*Sin(pi/18), "0"
            .Create
         End With
         Next j
    Next i

    
    End Sub"""
    return macros


def create_circular_macros(coils):
    rad = 'Array(' + ', '.join(f'Array({", ".join(str(r) for r in coil)})' for coil in coils) + ')'
    macros = f"""Option Explicit

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


	
End Sub"""
    return macros


with open('macros2.txt', 'w') as f:
    f.write(create_circular_macros(coils))


def create_rectangular_macros(coils):
    coil_a = 'Array(' + ', '.join(f'Array({", ".join(str(r) for r in coil)})' for coil in coils) + ')'
    coil_b = 'Array(' + ', '.join(f'Array({", ".join(str(r) + "/2" for r in coil)})' for coil in coils) + ')'

    macros = f"""

Sub Main ()


Dim coil_a() As Variant, coil_b() As Variant, a As Variant, b As Variant, i As Integer, j As Integer, d_wire As Variant, gap As Variant

coil_a = {coil_a}
coil_b = {coil_b}
gap = 0.01
d_wire = 0.001

For i = 0 To UBound(coil_a)
		For j = 0 To UBound(coil_a(i))
			a = coil_a(i)(j)
			b = coil_b(i)(j)
			With Polygon
				.Reset
				.Name "turn" + Str(i) + Str(j)
				.Curve "curve1"
				.Point a/2, 0
				.LineTo a/2, -b/2
				.LineTo -a/2, -b/2
				.LineTo -a/2, b/2
				.lineTo a/2, b/2
				.LineTo a/2, gap
				.Create
			End With
		Next j
Next i



For i = 0 To UBound(coil_a)
		For j = 0 To UBound(coil_a(i)) - 1
			With Polygon3D
				.Reset
				.Version 10
				.Name "connection" + Str(i) + Str(j)
				.Curve "curve1"
				.Point coil_a(i)(j)/2,0,0
				.Point coil_a(i)(j)/2,0,i*d_wire*2
				.Point coil_a(i)(j+1)/2,gap,i*d_wire*2
				.Point coil_a(i)(j+1)/2,gap,0
				.Create
			End With
		Next j
Next i


For i = 0 To UBound(coil_a)
		With Polygon3D
			.Reset
			.Version 10
			.Name "line" + Str(i) + Str(j)
			.Curve "curve1"
			.Point coil_a(i)(0)/2,gap,0
			.Point coil_a(i)(0)/2,gap,(UBound(coil_a)+1)*d_wire*2
			.Create
		End With
Next i

For i = 0 To UBound(coil_a)
		With Polygon3D
			.Reset
			.Version 10
			.Name "line2" + Str(i) + Str(j)
			.Curve "curve1"
			.Point coil_a(i)(UBound(coil_a(i)))/2, 0, 0
			.Point coil_a(i)(UBound(coil_a(i)))/2, 0, (UBound(coil_a)+1)*d_wire*2
			.Create
		End With
Next i




End Sub"""
    return macros


def create_square_macros(coils):
    coil_a = 'Array(' + ', '.join(f'Array({", ".join(str(r) for r in coil)})' for coil in coils) + ')'

    macros = f"""' macros

Sub Main ()


Dim coil_a() As Variant, a As Variant, i As Integer, j As Integer, d_wire As Variant, gap As Variant

coil_a = {coil_a}
gap = 0.01
d_wire = 0.001

For i = 0 To UBound(coil_a)
		For j = 0 To UBound(coil_a(i))
			a = coil_a(i)(j)
			With Polygon
				.Reset
				.Name "turn" + Str(i) + Str(j)
				.Curve "curve1"
				.Point a/2, 0
				.LineTo a/2, -a/2
				.LineTo -a/2, -a/2
				.LineTo -a/2, a/2
				.lineTo a/2, a/2
				.LineTo a/2, gap
				.Create
			End With
		Next j
Next i



For i = 0 To UBound(coil_a)
		For j = 0 To UBound(coil_a(i)) - 1
			With Polygon3D
				.Reset
				.Version 10
				.Name "connection" + Str(i) + Str(j)
				.Curve "curve1"
				.Point coil_a(i)(j)/2,0,0
				.Point coil_a(i)(j)/2,0,i*d_wire*2
				.Point coil_a(i)(j+1)/2,gap,i*d_wire*2
				.Point coil_a(i)(j+1)/2,gap,0
				.Create
			End With
		Next j
Next i


For i = 0 To UBound(coil_a)
		With Polygon3D
			.Reset
			.Version 10
			.Name "line" + Str(i) + Str(j)
			.Curve "curve1"
			.Point coil_a(i)(0)/2,gap,0
			.Point coil_a(i)(0)/2,gap,(UBound(coil_a)+1)*d_wire*2
			.Create
		End With
Next i

For i = 0 To UBound(coil_a)
		With Polygon3D
			.Reset
			.Version 10
			.Name "line2" + Str(i) + Str(j)
			.Curve "curve1"
			.Point coil_a(i)(UBound(coil_a(i)))/2, 0, 0
			.Point coil_a(i)(UBound(coil_a(i)))/2, 0, (UBound(coil_a)+1)*d_wire*2
			.Create
		End With
Next i




End Sub"""
    return macros

# with open('macros.mcs', 'wb') as f:
#     f.write()
