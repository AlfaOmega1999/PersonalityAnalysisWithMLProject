import React from 'react'
import {
    Nav,
    NavLink,
    Bars,
    NavMenu,
    NavBtn,
    NavBtnLink
  } from './NavbarElements';
  import logo from '../../images/MBTILogo.png';

const Navbar = () => {
  return (
    <>
      <Nav>
          <NavLink to="/">
          <img src={logo} className="App-logo" alt="logo" />
          </NavLink>
          <Bars />
          <NavMenu>
              <NavLink to="/types" >
                  Types
              </NavLink>
              <NavLink to="/stats" >
                  Stats
              </NavLink>
              <NavLink to="/about" >
                  About
              </NavLink>
          </NavMenu>
          <NavBtn>
              <NavBtnLink to="/predictor">Predictor</NavBtnLink>
          </NavBtn>
      </Nav>
    </>
  )
}

export default Navbar
