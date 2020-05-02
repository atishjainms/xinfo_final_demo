import { Component, OnInit } from '@angular/core';
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';
import {HomeServiceService} from '../home-service.service'


@Component({
  selector: 'app-first',
  templateUrl: './first.component.html',
  styleUrls: ['./first.component.css']
})
export class FirstComponent implements OnInit {
  private displayvalues
  private searchText
  private showText
  private displayvaluesflag
  private showTextFlag
  constructor(private homeService:HomeServiceService){

  }
  ngOnInit() {
    this.displayvaluesflag=false;
    this.showTextFlag=false
}
viewText(d){
  this.showTextFlag=true
  this.showText=d.text
}

  query(): void {
 this.homeService.getNewsInformation()
   .subscribe((data) => {this.displayvalues=data}


   )
  this.displayvaluesflag=true
}
}
