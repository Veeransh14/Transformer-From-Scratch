from __future__ import annotations
import numpy as onp
import jax.numpy as np
import math
import functools


def inv_sqrt(x):
  return 1.0 / np.sqrt(x)

def _data(v, dtype):
  # #if isinstance(v, np.ndarray):
  # #  return v
  # #if isinstance(v, MData):
  # #  return v._data
  # if hasattr(v, '_data'):
  #   return v.astype(dtype)
  # if isinstance(v, (tuple, list)):
  #   return np.asarray(v, dtype=dtype)
  # return v
  return np.asarray(v, dtype=dtype)

def _data_assign(lhs, rhs):
  if rhs.ndim == 1:
    lhs[0:len(rhs)] = rhs[:]
  else:
    for i in range(len(rhs)):
      _data_assign(lhs[i], rhs[i])

@functools.total_ordering
class MData:
  def __init__(self, data):
    self._data = data

  def __jax_array__(self):
    return self._data

  def __matmul__(self, v): return self.__class__(self._data @ _data(v, self._data.dtype))

  def __add__(self, v): return self.__class__(self._data + _data(v, self._data.dtype))
  def __sub__(self, v): return self.__class__(self._data - _data(v, self._data.dtype))
  def __mul__(self, v): return self.__class__(self._data * _data(v, self._data.dtype))
  def __radd__(self, v): return self.__class__(self._data + _data(v, self._data.dtype))
  def __rsub__(self, v): return self.__class__(self._data - _data(v, self._data.dtype))
  def __rmul__(self, v): return self.__class__(self._data * _data(v, self._data.dtype))
  def __neg__(self): return self.__class__(- self._data)
  def __pos__(self): return self.__class__(+ self._data)
  def __abs__(self): return self.__class__(np.abs(self._data))
  #def __trunc__(self): return self.__class__(math.trunc(self._data))

  def __lt__(self, other):
    return np.less(self._data, _data(other, self._data.dtype)).all()

  def __eq__(self, other):
    return np.equal(self._data, _data(other, self._data.dtype)).all()

  def __getitem__(self, i):
    return self._data[i]

  def __setitem__(self, i, value):
    self._data = self._data.at[i].set(value)

  def assign(self, rhs):
    self[:] = _data(rhs, self._data.dtype)[:]
    #_data_assign(self._data, _data(rhs, self._data.dtype))

  @property
  def data(self):
    return self._data[:]

  def astype(self, dtype):
    return self._data.astype(dtype)


class MVec3(MData):
  def __init__(self, *args):
    super().__init__(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    if len(args) == 1:
      self[:] = _data(args[0], self._data.dtype)
    elif len(args) == 3:
      self[0] = args[0]
      self[1] = args[1]
      self[2] = args[2]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  @staticmethod
  def from_xyz(x, y, z):
    return MVec3(x, y, z)

  @staticmethod
  def from_vec3(v):
    return MVec3(v.x, v.y, v.z)

  def __str__(self):
    return str(self._data)

  def __repr__(self):
    return '<MVec3 {}>'.format(repr(self._data))

  @property
  def x(self):
    return self._data[0]

  @x.setter
  def x(self, value):
    self[0] = value

  @property
  def y(self):
    return self._data[1]

  @y.setter
  def y(self, value):
    self[1] = value

  @property
  def z(self):
    return self._data[2]

  @z.setter
  def z(self, value):
    self[2] = value

  def dot(self, v):
    return np.dot(self._data, _data(v, self._data.dtype))

  @property
  def mag_sqr(self):
    x = self._data[0]
    y = self._data[1]
    z = self._data[2]
    return x*x + y*y + z*z

  @property
  def mag(self):
    return np.sqrt(self.mag_sqr)

  def normalize(self):
    invMag = 1.0 / self.mag
    self.x *= invMag
    self.y *= invMag
    self.z *= invMag

  def normalized(self):
    r = self.__class__(self)
    r.normalize()
    return r

  def cross(self, v):
    _data = self._data
    return MVec3( _data[ 1 ]*v._data[ 2 ] - _data[ 2 ]*v._data[ 1 ],
                  _data[ 2 ]*v._data[ 0 ] - _data[ 0 ]*v._data[ 2 ],
                  _data[ 0 ]*v._data[ 1 ] - _data[ 1 ]*v._data[ 0 ] )

  def invert(self):
    self.x = 1.0 / self.x
    self.y = 1.0 / self.y
    self.z = 1.0 / self.z

  def inverted(self):
    r = self.__class__(self)
    r.invert()
    return r


class MVec4(MData):
  def __init__(self, *args):
    super().__init__(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    if len(args) == 1:
      self[:] = _data(args[0], self._data.dtype)
    elif len(args) == 4:
      self[0] = args[0]
      self[1] = args[1]
      self[2] = args[2]
      self[3] = args[3]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  @staticmethod
  def from_xyzw(x, y, z, w):
    return MVec3(x, y, z, w)

  @staticmethod
  def from_vec4(v):
    return MVec4(v.x, v.y, v.z, v.w)

  def __str__(self):
    return str(self._data)

  def __repr__(self):
    return '<MVec4 {}>'.format(repr(self._data))

  @property
  def x(self):
    return self._data[0]

  @x.setter
  def x(self, value):
    self[0] = value

  @property
  def y(self):
    return self._data[1]

  @y.setter
  def y(self, value):
    self[1] = value

  @property
  def z(self):
    return self._data[2]

  @z.setter
  def z(self, value):
    self[2] = value

  @property
  def w(self):
    return self._data[3]

  @w.setter
  def w(self, value):
    self[3] = value


class MMat3x3(MData):
  def __init__(self, *args):
    super().__init__(np.eye(3))
    if len(args) == 1:
      v = args[0]
      self.assign(v)
    elif len(args) == 9:
      self[ 0, 0 ] = args[ 0 ]
      self[ 0, 1 ] = args[ 1 ]
      self[ 0, 2 ] = args[ 2 ]
      self[ 1, 0 ] = args[ 3 ]
      self[ 1, 1 ] = args[ 4 ]
      self[ 1, 2 ] = args[ 5 ]
      self[ 2, 0 ] = args[ 6 ]
      self[ 2, 1 ] = args[ 7 ]
      self[ 2, 2 ] = args[ 8 ]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  def __repr__(self):
    return "<MMat3x3\n{}>".format(self._data)

  def __getitem__(self, idx):
    r = super().__getitem__(idx)
    if isinstance(idx, int):
      r = MVec3(r)
    return r

  def get_at(self, i):
    return self._data[i // 3, i % 3]

  def set_at(self, i, value):
    self[i // 3, i % 3] = value

  def get_col(self, i):
    return MVec3(
        self._data[ 0,  i ],
        self._data[ 1,  i ],
        self._data[ 2,  i ])

  def set_col(self, i, col):
    col = _data( col, self._data.dtype )
    self[ 0, i ] = col[0]
    self[ 1, i ] = col[1]
    self[ 2, i ] = col[2]

  def transpose(self):
    self[0, 1], self[1, 0] = self._data[1, 0], self._data[0, 1]
    self[0, 2], self[2, 0] = self._data[2, 0], self._data[0, 2]
    self[0, 3], self[3, 0] = self._data[3, 0], self._data[0, 3]

  def transposed(self):
    return self.__class__(
        self._data[0, 0], self._data[1, 0], self._data[2, 0],
        self._data[0, 1], self._data[1, 1], self._data[2, 1],
        self._data[0, 2], self._data[1, 2], self._data[2, 2])

  # def inverse_transposed(self, output: MMat3x3 = None):
  #   if output is None:
  #     r = self.__class__()
  #     assert self.inverse_transposed(r)
  #     return r
  #   if not output.inverse(output):
  #     return False
  #   output.transpose()
  #   return True

  def inverse_transposed(self):
    return self.inverse().transposed()

  def __mul__(self, m):
    return self.__class__(np.matmul(self._data, _data(m, self._data.dtype)))

  def inverse(self, output=None):
    if output is None:
      r = self.__class__()
      assert self.inverse(r)
      return r
    try:
      m = np.linalg.inv(self._data)
      output.assign(m)
      return True
    except np.linalg.LinAlgError: # TODO: What's the JAX equivalent of this?
      return False

  def rotate_point(self, point):
    point = _data(point, self._data.dtype)
    return MVec3(
      self.get_at( 0 ) * point[0] + self.get_at( 1 ) * point[1] + self.get_at( 2 ) * point[2],
      self.get_at( 3 ) * point[0] + self.get_at( 4 ) * point[1] + self.get_at( 5 ) * point[2],
      self.get_at( 6 ) * point[0] + self.get_at( 7 ) * point[1] + self.get_at( 8 ) * point[2] )

  def rotate_point_fast(self, point):
    x = self.get_at( 0 ) * point.x + self.get_at( 1 ) * point.y + self.get_at( 2 ) * point.z
    y = self.get_at( 3 ) * point.x + self.get_at( 4 ) * point.y + self.get_at( 5 ) * point.z
    z = self.get_at( 6 ) * point.x + self.get_at( 7 ) * point.y + self.get_at( 8 ) * point.z
    point.x = x
    point.y = y
    point.z = z


class MMat4x4(MData):
  def __init__(self, *args):
    super().__init__(np.eye(4))
    if len(args) == 1:
      v = _data(args[0], self._data.dtype)
      if v.ndim == 2 and onp.prod(onp.shape(v)) == 9:
        rot = v
        self[ 0, 0 ] = rot[ 0, 0 ]
        self[ 0, 1 ] = rot[ 0, 1 ]
        self[ 0, 2 ] = rot[ 0, 2 ]

        self[ 1, 0 ] = rot[ 1, 0 ]
        self[ 1, 1 ] = rot[ 1, 1 ]
        self[ 1, 2 ] = rot[ 1, 2 ]

        self[ 2, 0 ] = rot[ 2, 0 ]
        self[ 2, 1 ] = rot[ 2, 1 ]
        self[ 2, 2 ] = rot[ 2, 2 ]
      else:
        self.assign(v)
    elif len(args) == 2:
      rot = _data(args[0], self._data.dtype)
      pos = _data(args[1], self._data.dtype)
      self[ 0, 0 ] = rot[ 0, 0 ]
      self[ 0, 1 ] = rot[ 0, 1 ]
      self[ 0, 2 ] = rot[ 0, 2 ]
      self[ 0, 3 ] = pos[ 0 ]

      self[ 1, 0 ] = rot[ 1, 0 ]
      self[ 1, 1 ] = rot[ 1, 1 ]
      self[ 1, 2 ] = rot[ 1, 2 ]
      self[ 1, 3 ] = pos[ 1 ]

      self[ 2, 0 ] = rot[ 2, 0 ]
      self[ 2, 1 ] = rot[ 2, 1 ]
      self[ 2, 2 ] = rot[ 2, 2 ]
      self[ 2, 3 ] = pos[ 2 ]

    elif len(args) == 16:
      self[ 0, 0 ] = args[ 0 ]
      self[ 0, 1 ] = args[ 1 ]
      self[ 0, 2 ] = args[ 2 ]
      self[ 0, 3 ] = args[ 3 ]

      self[ 1, 0 ] = args[ 4 ]
      self[ 1, 1 ] = args[ 5 ]
      self[ 1, 2 ] = args[ 6 ]
      self[ 1, 3 ] = args[ 7 ]

      self[ 2, 0 ] = args[ 8 ]
      self[ 2, 1 ] = args[ 9 ]
      self[ 2, 2 ] = args[ 10 ]
      self[ 2, 3 ] = args[ 11 ]

      self[ 3, 0 ] = args[ 12 ]
      self[ 3, 1 ] = args[ 13 ]
      self[ 3, 2 ] = args[ 14 ]
      self[ 3, 3 ] = args[ 15 ]
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  def __repr__(self):
    return "<MMat4x4\n{}>".format(self._data)

  def __getitem__(self, idx):
    r = super().__getitem__(idx)
    if isinstance(idx, int):
      r = MVec4(r)
    return r

  def get_at(self, i):
    return self[i // 4, i % 4]

  def set_at(self, i, value):
    self[i // 4, i % 4] = value

  def __mul__(self, m):
    return self.__class__(np.matmul(self._data, _data(m, self._data.dtype)))

  def inverse(self, output=None):
    if output is None:
      r = self.__class__()
      assert self.inverse(r)
      return r
    try:
      m = np.linalg.inv(self._data)
      output.assign(m)
      return True
    except np.linalg.LinAlgError: # TODO: What's the JAX equivalent of this?
      return False

  def get_rotate(self, mat):
    _data = self._data.reshape([-1])
    xInvScale = inv_sqrt( _data[ 0 ]*_data[ 0 ] + _data[ 1 ]*_data[ 1 ] + _data[ 2 ]*_data[ 2 ] )
    yInvScale = inv_sqrt( _data[ 4 ]*_data[ 4 ] + _data[ 5 ]*_data[ 5 ] + _data[ 6 ]*_data[ 6 ] )
    zInvScale = inv_sqrt( _data[ 8 ]*_data[ 8 ] + _data[ 9 ]*_data[ 9 ] + _data[ 10 ]*_data[ 10 ] )

    mat[ 0,0 ] = xInvScale*_data[ 0 ]
    mat[ 0,1 ] = xInvScale*_data[ 1 ]
    mat[ 0,2 ] = xInvScale*_data[ 2 ]

    mat[ 1,0 ] = yInvScale*_data[ 4 ]
    mat[ 1,1 ] = yInvScale*_data[ 5 ]
    mat[ 1,2 ] = yInvScale*_data[ 6 ]

    mat[ 2,0 ] = zInvScale*_data[ 8 ]
    mat[ 2,1 ] = zInvScale*_data[ 9 ]
    mat[ 2,2 ] = zInvScale*_data[ 10 ]

  @property
  def rotate(self):
    _data = self._data.reshape([-1])
    xInvScale = inv_sqrt( _data[ 0 ]*_data[ 0 ] + _data[ 1 ]*_data[ 1 ] + _data[ 2 ]*_data[ 2 ] )
    yInvScale = inv_sqrt( _data[ 4 ]*_data[ 4 ] + _data[ 5 ]*_data[ 5 ] + _data[ 6 ]*_data[ 6 ] )
    zInvScale = inv_sqrt( _data[ 8 ]*_data[ 8 ] + _data[ 9 ]*_data[ 9 ] + _data[ 10 ]*_data[ 10 ] )
    return MMat3x3( xInvScale*_data[ 0 ], xInvScale*_data[ 1 ], xInvScale*_data[ 2 ],
                    yInvScale*_data[ 4 ], yInvScale*_data[ 5 ], yInvScale*_data[ 6 ],
                    zInvScale*_data[ 8 ], zInvScale*_data[ 9 ], zInvScale*_data[ 10 ] )

  @rotate.setter
  def rotate(self, rot):
    scale = self.scale
    self[ 0, 0 ] = scale.x * rot[ 0, 0 ]
    self[ 0, 1 ] = scale.x * rot[ 0, 1 ]
    self[ 0, 2 ] = scale.x * rot[ 0, 2 ]
    self[ 1, 0 ] = scale.y * rot[ 1, 0 ]
    self[ 1, 1 ] = scale.y * rot[ 1, 1 ]
    self[ 1, 2 ] = scale.y * rot[ 1, 2 ]
    self[ 2, 0 ] = scale.z * rot[ 2, 0 ]
    self[ 2, 1 ] = scale.z * rot[ 2, 1 ]
    self[ 2, 2 ] = scale.z * rot[ 2, 2 ]

  def get_translate(self, pos):
    _data = self._data.reshape([-1])
    pos.x = _data[ 3 ]
    pos.y = _data[ 7 ]
    pos.z = _data[ 11 ]

  @property
  def translate(self):
    _data = self._data.reshape([-1])
    return MVec3( _data[ 3 ], _data[ 7 ], _data[ 11 ] )

  @translate.setter
  def translate(self, pos):
    self[ 0, 3 ] = pos.x
    self[ 1, 3 ] = pos.y
    self[ 2, 3 ] = pos.z

  def get_scale_sqr(self, scale):
    # extract the scale of the matrix.  Ensure that the rotational component of the matrix is of uniform scaling.
    _data = self._data.reshape([-1])
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )

    # return the scales of the axes.
    scale.x = x.mag_sqr
    scale.y = y.mag_sqr
    scale.z = z.mag_sqr

  @property
  def scale_sqr(self):
    # extract the scale of the matrix.  Ensure that the rotational component of the matrix is of uniform scaling.
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )

    # return the scales of the axes.
    return MVec3( x.mag_sqr, y.mag_sqr, z.mag_sqr )


  def get_scale(self, scale):
    # extract the scale of the matrix.  Ensure that the rotational component of the matrix is of uniform scaling.
    _data = self._data.reshape([-1])
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )

    # return the scales of the axes.
    scale.x = x.mag
    scale.y = y.mag
    scale.z = z.mag

  @property
  def scale(self):
    _data = self._data.reshape([-1])
    # extract the scale of the matrix.  Ensure that the rotational component of the matrix is of uniform scaling.
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )

    # return the scales of the axes.
    return MVec3( x.mag, y.mag, z.mag )

  @scale.setter
  def scale(self, scale):
    _data = self._data.reshape([-1])
    # orthonormalize the matrix axes.
    x = MVec3( _data[ 0 ], _data[ 1 ], _data[ 2 ] )
    y = MVec3( _data[ 4 ], _data[ 5 ], _data[ 6 ] )
    z = MVec3( _data[ 8 ], _data[ 9 ], _data[ 10 ] )
    x.normalize()
    y.normalize()
    z.normalize()
    x = x * scale.x
    y = y * scale.y
    z = z * scale.z
    self.set_axes(x, y, z)

  def set_axes(self, x, y, z):
    self[ 0, 0 ] = x.x
    self[ 0, 1 ] = x.y
    self[ 0, 2 ] = x.z
    self[ 1, 0 ] = y.x
    self[ 1, 1 ] = y.y
    self[ 1, 2 ] = y.z
    self[ 2, 0 ] = z.x
    self[ 2, 1 ] = z.y
    self[ 2, 2 ] = z.z

  def set_orientation(self, side, forward):
    # compute the new matrix axes.
    x = MVec3( side.normalized() )
    z = MVec3( forward.normalized() )
    y = MVec3( z.cross( x ).normalized() )
    x = y.cross( z ).normalized()

    # scale them.
    scale = self.scale
    x = x * scale.x
    y = y * scale.y
    z = z * scale.z

    # set them.
    self.set_axes( x, y, z )

  def transform_coord_no_persp(self, coord):
    coord = _data(coord, self._data.dtype)
    x = self.get_at( 0 ) * coord[0] + self.get_at( 1 ) * coord[1] + self.get_at(  2 ) * coord[2] + self.get_at( 3 )
    y = self.get_at( 4 ) * coord[0] + self.get_at( 5 ) * coord[1] + self.get_at(  6 ) * coord[2] + self.get_at( 7 )
    z = self.get_at( 8 ) * coord[0] + self.get_at( 9 ) * coord[1] + self.get_at( 10 ) * coord[2] + self.get_at( 11 )
    return MVec3( x, y, z )

  def transform_coord_no_persp_fast(self, coord):
    x = self.get_at( 0 ) * coord[0] + self.get_at( 1 ) * coord[1] + self.get_at(  2 ) * coord[2] + self.get_at( 3 )
    y = self.get_at( 4 ) * coord[0] + self.get_at( 5 ) * coord[1] + self.get_at(  6 ) * coord[2] + self.get_at( 7 )
    z = self.get_at( 8 ) * coord[0] + self.get_at( 9 ) * coord[1] + self.get_at( 10 ) * coord[2] + self.get_at( 11 )
    coord.x = x
    coord.y = y
    coord.z = z


class MPlane:
  def __init__(self, *args):
    self._normal = MVec3(0.0, 1.0, 0.0)
    self._d = 0.0
    if len(args) == 1:
      v = args[0]
      if isinstance(v, MPlane):
        self._normal[:] = v._normal[:]
        self._d = v._d
      else:
        raise ValueError("Unknown argument type {}".format(v))
    elif len(args) == 2:
      #if not isinstance(args[0], MVec3):
      #  raise ValueError("Argument 0: expected MVec3, got {}".format(args[0]))
      self._normal.assign(args[0])
      self._normal.normalize()
      if isinstance(args[1], float) or isinstance(args[1], int):
        self._d = args[1]
      else:
        point = _data(args[1], self._normal._data.dtype)
        self._d = -self._normal.dot(point)
    elif len(args) != 0:
      raise ValueError("Bad arguments: {}".format(args))

  def assign(self, rhs):
    self._normal.assign(rhs._normal)
    self._d = rhs._d

  def __repr__(self):
    return "<MPlane normal=({}, {}, {}) d={}>".format(self._normal.x, self._normal.y, self._normal.z, self._d)

  @property
  def normal(self):
    return self._normal

  @normal.setter
  def normal(self, value):
    self._normal.assign(value)

  @property
  def d(self):
    return self._d

  @d.setter
  def d(self, value):
    assert isinstance(value, float) or isinstance(value, int)
    self._d = float(value)

  def dist(self, point):
    return self._normal.dot(point) + self._d


