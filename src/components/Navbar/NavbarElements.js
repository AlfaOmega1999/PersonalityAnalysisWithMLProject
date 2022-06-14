import styled from'styled-components'
import {NavLink as Link} from 'react-router-dom'
import {FaBars} from 'react-icons/fa'

export const Nav = styled.nav`
  background: #091222;
  border-style: hidden hidden solid hidden;
  border-color: #BE9533;
  height: 80px;
  display: flex;
  justify-content: space-between;
  padding: 0.5rem calc((100vw - 1000px) / 2);
  z-index: 10;
  /* Third Nav */
  /* justify-content: flex-start; */
`;
export const Nav2 = styled.nav`
  height: 80px;
  display: flex;
  justify-content: space-between;
  z-index: 10;
  /* Third Nav */
  /* justify-content: flex-start; */
`;
export const NavLink = styled(Link)`
  color: #fff;
  display: flex;
  align-items: center;
  text-decoration: none;
  padding: 0 1rem;
  font-size: 25px;
  font-weight: bold;
  height: 100%;
  cursor: pointer;
  &.active {
    color: #BE9533;
    -webkit-text-shadow: 0 0 15px #BE9533;
            text-shadow: 0 0 15px #BE9533;
  }
  &:hover {
    color: #BE9533;
	-webkit-text-shadow: 0 0 15px #BE9533;
          text-shadow: 0 0 15px #BE9533;
  }
`;

export const Bars = styled(FaBars)`
  display: none;
  color: #fff;
  @media screen and (max-width: 768px) {
    display: block;
    position: absolute;
    top: 0;
    right: 0;
    transform: translate(-100%, 75%);
    font-size: 1.8rem;
    cursor: pointer;
  }
`;

export const NavMenu = styled.div`
  display: flex;
  align-items: center;
  margin-right: -24px;
  /* Second Nav */
  /* margin-right: 24px; */
  /* Third Nav */
  /* width: 100vw;
  white-space: nowrap; */
  @media screen and (max-width: 768px) {
    display: none;
  }
`;

export const NavBtn = styled.nav`
  display: flex;
  align-items: center;
  font-size: 20px;
  font-weight: bold;
  margin-right: 24px;
  /* Third Nav */
  /* justify-content: flex-end;
  width: 100vw; */
  @media screen and (max-width: 768px) {
    display: none;
  }
`;

export const NavBtnLink = styled(Link)`
  border-radius: 4px;
  background: #BE9533;
  background: -webkit-linear-gradient(to right, #F8CE4D, #534714);  /* Chrome 10-25, Safari 5.1-6 */
	background: -webkit-gradient(linear, left top, right top, from(#B39D45), to(#534714));
	background: -webkit-linear-gradient(left, #B39D45, #F8CE4D);
	background: -o-linear-gradient(left, #B39D45, #534714);
	background: linear-gradient(to right, #B39D45, #534714);
  padding: 10px 22px;
  color: #fff;
  outline: none;
  border: none;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  text-decoration: none;
  /* Second Nav */
  margin-left: 24px;
  &:hover {
    transition: all 0.2s ease-in-out;
    background: #fff;
    color: #BE9533; 
  -webkit-box-shadow: 0 0 20px #BE9533;
          box-shadow: 0 0 20px #BE9533;
  }
`;