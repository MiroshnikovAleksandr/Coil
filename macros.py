coils = [[0.08, 0.06, 0.04, 0.02]]
coords = [[[[-1.0, 0.0], [-0.5, 0.866025], [0.5, 0.866025], [1.0, 0.0], [0.5, -0.866025], [-0.5, -0.866025]]], [
    [[-0.13333333333333333, 0.0], [-0.06666666666666667, 0.11547], [0.06666666666666667, 0.11547],
     [0.13333333333333333, 0.0], [0.06666666666666667, -0.11547], [-0.06666666666666667, -0.11547]],
    [[-0.06666666666666667, 0.0], [-0.03333333333333333, 0.057735], [0.03333333333333333, 0.057735],
     [0.06666666666666667, 0.0], [0.03333333333333333, -0.057735], [-0.03333333333333333, -0.057735]],
    [[-0.03333333333333333, 0.0], [-0.016666666666666666, 0.0288675], [0.016666666666666666, 0.0288675],
     [0.03333333333333333, 0.0], [0.016666666666666666, -0.0288675], [-0.016666666666666666, -0.0288675]]]]


def create_piecewise_macros(coords):
    s_2 = ''
    for subcoil in coords:
        s_1 = ''
        for turn in subcoil:
            turn = 'Array(' + ', '.join(f'Array({", ".join(str(num) for num in pair)})' for pair in turn) + ')'
            s_1 += turn + ', '
        s_1 = s_1[:-2]
        coil = 'Array(' + s_1 + ')'
        s_2 += coil + ', '
    s_2 = s_2[:-2]
    rad = 'Array(' + s_2 + ')'

    macros = f"""
Sub Main
    
    
Dim rad() As Variant, r As Variant, i As Integer, j As Integer, k As Integer, gap As Variant
    rad = {rad}
    gap = 0.03
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


    For i = 0 To UBound(rad)
        For j = UBound(rad(i)) To 1 STEP -1
        	r = rad(i)(j)
        	With Line
              .Reset
              .Name "connection" + Str(j) + " -" + Str(j - 1)
              .Curve "curve1"
              .X1 r(0)(0)
              .Y1 r(0)(1) - gap
              .X2 rad(i)(j-1)(0)(0)
              .Y2 rad(i)(j-1)(0)(1)
              .Create
            End With
        Next j
    Next i
    
    
End Sub
"""
    return macros


with open('macros1.txt', 'w') as f:
    f.write(create_piecewise_macros(coords))


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
