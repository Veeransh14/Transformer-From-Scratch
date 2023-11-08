from ..common import math as m

import jax.numpy as np

def deg_to_rad(deg):
  return deg * np.pi / 180.0

class GrProjection:
  def __init__(self):
    self._fov = 0.0
    self._near_clip = m.MPlane( m.MVec3( 0.0, 0.0, -1.0 ), m.MVec3( 0.0, 0.0, -1.0 ) )
    self._far_dist = 1000.0
    self._aspect = 1.0
    self._left = -1.0
    self._right = 1.0
    self._bottom = -1.0
    self._top = 1.0
    self._mat = m.MMat4x4()
    self._dirty = True

  def assign(self, rhs):
    self._fov = rhs._fov
    self._near_clip.assign(rhs._near_clip)
    self._far_dist = rhs._far_dist
    self._aspect = rhs._aspect
    self._left = rhs._left
    self._right = rhs._right
    self._bottom = rhs._bottom
    self._top = rhs._top
    self._mat.assign(rhs._mat)
    self._dirty = rhs._dirty

  @classmethod
  def perspective(cls, fov, far_dist, aspect, near_clip_plane):
    self = cls()
    self._fov = fov
    self._far_dist = far_dist
    self._aspect = aspect
    self._near_clip.assign(near_clip_plane)
    assert self._fov >= 0.0
    return self

  def __repr__(self):
    return "<GrProjection fov={} near_clip={} far_dist={} aspect={} left={} right={} bottom={} top={}>".format(self._fov, self._near_clip, self._far_dist, self._aspect, self._left, self._right, self._bottom, self._top)

  @property
  def is_ortho(self):
    return self._fov == 0.0

  @property
  def fov(self):
    return self._fov

  @fov.setter
  def fov(self, value):
    self._dirty = True
    self._fov = value

  @property
  def width(self):
    return self._right - self._left

  @property
  def height(self):
    return self._top - self._bottom

  @property
  def near_clip(self):
    return self._near_clip

  @near_clip.setter
  def near_clip(self, value):
    self._dirty = True
    self._near_clip = value

  @property
  def far_dist(self):
    return self._far_dist

  @far_dist.setter
  def far_dist(self, value):
    self._dirty = True
    self._far_dist = value

  @property
  def aspect(self):
    return self._aspect

  @aspect.setter
  def aspect(self, value):
    self._dirty = True
    self._aspect = value

  @property
  def left(self):
    return self._left

  @left.setter
  def left(self, value):
    self._dirty = True
    self._left = value

  @property
  def right(self):
    return self._right

  @right.setter
  def right(self, value):
    self._dirty = True
    self._right = value

  @property
  def top(self):
    return self._top

  @top.setter
  def top(self, value):
    self._dirty = True
    self._top = value

  @property
  def bottom(self):
    return self._bottom

  @bottom.setter
  def bottom(self, value):
    self._dirty = True
    self._bottom = value

  @property
  def matrix(self):
    if self._dirty:
      assert abs(self._right - self._left) > 0.0001
      assert abs(self._top - self._bottom) > 0.0001

      xScale = 2.0 / ( self._right - self._left )
      yScale = 2.0 / ( self._top - self._bottom )
      xOffset = ( self._right + self._left ) / ( self._right - self._left )
      yOffset = ( self._top + self._bottom ) / ( self._top - self._bottom )

      if not self.is_ortho:
        projNear = 1.0 / np.tan( self._fov * 0.5 )

        self._mat[ 0, 0 ] = xScale * projNear / ( self._aspect )
        self._mat[ 0, 1 ] = 0.0
        self._mat[ 0, 2 ] = xOffset
        self._mat[ 0, 3 ] = 0.0

        self._mat[ 1, 0 ] = 0.0
        self._mat[ 1, 1 ] = yScale * projNear
        self._mat[ 1, 2 ] = yOffset
        self._mat[ 1, 3 ] = 0.0

        self._mat[ 2, 0 ] = 0.0
        self._mat[ 2, 1 ] = 0.0
        self._mat[ 2, 2 ] = -1.0 # ( projNear + _farDist ) / ( projNear - _farDist )
        self._mat[ 2, 3 ] = -1.0 # _farDist * projNear / ( projNear - _farDist )

        self._mat[ 3, 0 ] = 0.0
        self._mat[ 3, 1 ] = 0.0
        self._mat[ 3, 2 ] = -1.0
        self._mat[ 3, 3 ] = 0.0

        # incorperate the near clip plane into the matrix by applying a shear
        # on the Z axis.
        self.add_near_clip()
      else:
        assert self._far_dist > 0.01

        self._mat[ 0, 0 ] = xScale
        self._mat[ 0, 1 ] = 0.0
        self._mat[ 0, 2 ] = 0.0
        self._mat[ 0, 3 ] = -xOffset

        self._mat[ 1, 0 ] = 0.0
        self._mat[ 1, 1 ] = yScale
        self._mat[ 1, 2 ] = 0.0
        self._mat[ 1, 3 ] = -yOffset

        self._mat[ 2, 0 ] = 0.0
        self._mat[ 2, 1 ] = 0.0
        self._mat[ 2, 2 ] = ( -1.0 / self._far_dist)
        self._mat[ 2, 3 ] = 0.0

        self._mat[ 3, 0 ] = 0.0
        self._mat[ 3, 1 ] = 0.0
        self._mat[ 3, 2 ] = 0.0
        self._mat[ 3, 3 ] = 1.0

      # clear the dirty flag.
      self._dirty = False

    # return the cached matrix.
    return self._mat

  def add_near_clip(self):
    clipNormal = self._near_clip.normal
    self._mat[ 2, 0 ] = clipNormal.x
    self._mat[ 2, 1 ] = clipNormal.y
    self._mat[ 2, 2 ] = clipNormal.z
    self._mat[ 2, 3 ] = self._near_clip.d


class GrCamera:
  def __init__(self):
    self._proj = GrProjection.perspective( deg_to_rad(90.0), 1000.0, 1.0, m.MPlane( m.MVec3( 0.0, 0.0, -1.0 ), m.MVec3( 0.0, 0.0, -1.0 ) ) )
    self._pos = m.MVec3()
    self._rot = m.MMat3x3()
    self._far_cull = 1000.0
    self._view_proj_matrix = m.MMat4x4()
    self._view_matrix = m.MMat4x4()
    self._inv_view_matrix = m.MMat4x4()
    self._normal_rot = m.MMat3x3()
    #self._frustum = GrFrustum()
    self._dirty = True

  def assign(self, rhs):
    self.pos = rhs.pos
    self.rot = rhs.rot
    self.proj = rhs.proj

  @property
  def pos(self):
    return self._pos

  @pos.setter
  def pos(self, pos):
    self._pos.assign(pos)
    self._dirty = True

  @property
  def rot(self):
    return self._rot

  @rot.setter
  def rot(self, rot):
    self._rot.assign(rot)
    self._dirty = True

  @property
  def proj(self):
    return self._proj

  @proj.setter
  def proj(self, proj):
    self._proj.assign(proj)
    self._dirty = True

  @property
  def side_dir(self):
    return self._rot.get_col(0)

  @property
  def up_dir(self):
    return self._rot.get_col(1)

  @property
  def look_dir(self):
    return self._rot.get_col(2)

  def look_at(self, pos, target, world_up=None):
    pos = m.MVec3(pos)
    target = m.MVec3(target)
    zdir = pos - target
    assert zdir.mag_sqr > 0.00001

    # mark as dirty.
    self._dirty = True

    # store the position.
    self._pos.assign(pos)

    # compute the new z basis.
    self.look(zdir, world_up=world_up)

  def look(self, tainted_dir, world_up=None):
    if world_up is None:
      world_up = (0.0, 1.0, 0.0)
    world_up = m.MVec3( world_up )
    world_up.normalize() # TODO: is this necessary?

    # mark as dirty.
    self._dirty = True

    # build the rotation matrix.
    look = tainted_dir.normalized()

    # compute the new x basis.
    if abs( look.y ) > 0.999:
      # compensate for trying to look directly up or down.
      right = m.MVec3(1, 0, 0)
    else:
      right = look.cross( world_up )
      right.normalize()

    # compute the new y basis.
    up = right.cross( look )

    # set the basis vectors.
    self._rot.set_col( 0, right )
    self._rot.set_col( 1, up )
    self._rot.set_col( 2, -look )

  def update_matrices(self):
    assert self._dirty

    # mark the matrices as up to date.
    self._dirty = False

    # create the projection matrix.
    projMatrix = self._proj.matrix

    transViewRot = self._rot.transposed()

    # create the view matrix.
    invCamPos = -transViewRot.rotate_point( self._pos )
    self._view_matrix = m.MMat4x4( transViewRot, invCamPos )

    # adjoint transpose.
    self._normal_rot = m.MMat3x3( transViewRot )

    # create the view-proj matrix.
    self._view_proj_matrix = projMatrix * self._view_matrix
    valid = self._view_matrix.inverse( self._inv_view_matrix )
    assert valid

    # reflection = False
    # # check to see if the view matrix is a reflection.
    # #if self._view_matrix[ 0, 0 ] * self._view_matrix[ 1, 1 ] * self._view_matrix[ 2, 2 ] < 0.0:
    # #  reflection = True;

    # # check to see if the view matrix is a reflection.
    # xAxis = m.MVec3( self._view_matrix[0][0:3] )
    # yAxis = m.MVec3( self._view_matrix[1][0:3] )
    # zAxis = m.MVec3( self._view_matrix[2][0:3] )

    # # do we have a reflection?
    # if xAxis.cross( yAxis ).dot( zAxis ) < 0.0:
    #   reflection = True;

  def build_world_matrix(self):
    return m.MMat4x4( self.rot, self.pos )

  @property
  def view_matrix(self):
    if self._dirty:
      self.update_matrices()
    return self._view_matrix

  @property
  def inv_view_matrix(self):
    if self._dirty:
      self.update_matrices()
    return self._inv_view_matrix

  @property
  def view_proj_matrix(self):
    if self._dirty:
      self.update_matrices()
    return self._view_proj_matrix

  @property
  def normal_matrix(self):
    if self._dirty:
      self.update_matrices()
    return self._normal_rot

